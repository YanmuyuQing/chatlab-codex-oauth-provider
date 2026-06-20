/**
 * Electron Agent stream runner — provides runAgentStream implementation
 * for the shared HTTP route context.
 *
 * Mirrors the logic from ipc/ai.ts `agent:runStream` handler,
 * but outputs events via callback instead of IPC.
 */

import type { AgentStreamChunk as SharedAgentStreamChunk } from '@openchatlab/node-runtime'
import type { AgentStreamRequest } from '@openchatlab/http-routes'
import type { ChartAutoMode } from '@openchatlab/shared-types'
import {
  buildSkillMenuWithBuiltinChart,
  buildCodexCliChatContext,
  CHART_CAPABILITY_ANALYSIS_TOOLS,
  checkAndCompress,
  CodexCliError,
  createCodexCliCompressionAdapter,
  createCompressionLlmAdapter,
  createDataSnapshotFromOverview,
  formatAIError,
  formatCodexCliChatContextError,
  formatCodexCliError,
  getCodexCliRetryMessageLimit,
  getCodexCliToolFallback,
  getAllowedBuiltinToolsForChartAutoSkill,
  getChartCapabilitySkill,
  initTokenizer,
  isCodexCliProvider,
  manualCompress,
  resolveChartRuntimeForRequest,
  runCodexCli,
} from '@openchatlab/node-runtime'
import type { CompressionConfig, CompressionLlmAdapter, AgentRuntimeStatus } from '@openchatlab/node-runtime'
import { Agent, type AgentStreamChunk, type SkillContext } from './agent'
import { buildSystemPrompt } from './agent/prompt-builder'
import { getAllTools } from './tools'
import type { ToolContext } from './tools/types'
import { getDefaultAssistantConfig, buildPiModel, findModelDefinition } from './llm'
import type { AIServiceConfig } from './llm/types'
import { getDefaultGeneralAssistantId } from './assistant/defaultGeneral'
import * as assistantManager from './assistant'
import type { AssistantConfig } from './assistant/types'
import * as skillManager from './skills'
import { aiLogger } from './logger'
import { serializeError } from './serialize-error'
import { getHistoryForAgent, getManager as getAIChatManager } from './chats'
import { t } from '../i18n'
import * as workerManager from '../worker/workerManager'
import { getProviderInfo, type LLMProvider } from './llm'

const DEFAULT_CONTEXT_WINDOW = 128000

function resolveProviderName(provider?: LLMProvider): string {
  if (provider === 'openai-compatible') return t('llm.genericProviderName')
  return provider ? getProviderInfo(provider)?.name || provider : t('llm.genericProviderName')
}

function buildCompressionAdapter(
  activeAIConfig: AIServiceConfig,
  onCompressing?: () => void,
  abortSignal?: AbortSignal
): CompressionLlmAdapter {
  if (isCodexCliProvider(activeAIConfig.provider)) {
    return createCodexCliCompressionAdapter({
      abortSignal,
      onCompressing,
      onError: (error) =>
        aiLogger.warn('Compression', 'Codex CLI compression attempt failed', { error: String(error) }),
    })
  }

  const modelDef = findModelDefinition(activeAIConfig.provider, activeAIConfig.model || '')
  return createCompressionLlmAdapter({
    piModel: buildPiModel(activeAIConfig),
    apiKey: activeAIConfig.apiKey,
    contextWindow: modelDef?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    onCompressing,
    onError: (error) => aiLogger.warn('Compression', 'LLM compression attempt failed', { error: String(error) }),
  })
}

const compressionLogger = {
  info: (cat: string, msg: string, extra?: Record<string, unknown>) => aiLogger.info(cat, msg, extra),
  warn: (cat: string, msg: string, extra?: Record<string, unknown>) => aiLogger.warn(cat, msg, extra),
  error: (cat: string, msg: string, extra?: Record<string, unknown>) => aiLogger.error(cat, msg, extra),
}

export function createElectronRunAgentStream(): (
  params: AgentStreamRequest,
  onEvent: (chunk: SharedAgentStreamChunk) => void,
  abortSignal: AbortSignal
) => Promise<void> {
  return async (params, onEvent, abortSignal) => {
    const {
      userMessage,
      aiChatId,
      historyLeafMessageId,
      sessionId,
      chatType,
      locale,
      assistantId,
      skillId,
      enableAutoSkill,
      chartAutoMode,
      compressionConfig,
      ownerInfo,
      mentionedMembers,
      thinkingLevel,
    } = params

    // 确保 tokenizer rank 表已加载（compression + agent 路径均依赖）
    await initTokenizer()

    const requestId = `internal_${Date.now()}`

    aiLogger.info('AgentStream', `Agent stream request: ${requestId}`, {
      userMessage: userMessage.slice(0, 100),
      sessionId,
      aiChatId,
      chatType: chatType ?? 'group',
      assistantId: assistantId ?? '(none)',
      skillId: skillId ?? '(none)',
      enableAutoSkill: enableAutoSkill ?? false,
    })

    const activeAIConfig = getDefaultAssistantConfig()
    if (!activeAIConfig) {
      onEvent({ type: 'error', error: { name: 'ConfigError', message: t('llm.notConfigured') } })
      return
    }
    if (compressionConfig?.enabled && aiChatId && historyLeafMessageId === undefined) {
      try {
        const tempAssistantConfig = assistantId
          ? (assistantManager.getAssistantConfig(assistantId) ?? undefined)
          : undefined
        const systemPromptForCompression = tempAssistantConfig?.systemPrompt || ''

        const compressionResult = await checkAndCompress(
          aiChatId,
          compressionConfig as CompressionConfig,
          systemPromptForCompression,
          buildCompressionAdapter(
            activeAIConfig,
            () => {
              onEvent({
                type: 'status',
                status: {
                  phase: 'compressing',
                  round: 0,
                  toolsUsed: 0,
                  contextTokens: 0,
                  totalUsage: {
                    promptTokens: 0,
                    completionTokens: 0,
                    totalTokens: 0,
                    cacheReadTokens: 0,
                    cacheWriteTokens: 0,
                  },
                  updatedAt: Date.now(),
                } satisfies AgentRuntimeStatus,
              })
            },
            abortSignal
          ),
          getAIChatManager(),
          compressionLogger
        )

        if (compressionResult.compressed && compressionResult.summaryContent) {
          onEvent({
            type: 'compression_done',
            compressionResult: {
              summaryContent: compressionResult.summaryContent,
              tokensBefore: compressionResult.tokensBefore ?? 0,
              tokensAfter: compressionResult.tokensAfter ?? 0,
              timestamp: Date.now(),
            },
          })
        }
      } catch (error) {
        aiLogger.error('AgentStream', `Compression failed: ${requestId}`, { error: String(error) })
      }
    }

    const defaultAssistantId = getDefaultGeneralAssistantId(locale)
    let resolvedAssistantId = assistantId || defaultAssistantId
    let assistantConfig: AssistantConfig | undefined =
      assistantManager.getAssistantConfig(resolvedAssistantId) ?? undefined
    if (!assistantConfig && resolvedAssistantId !== defaultAssistantId) {
      resolvedAssistantId = defaultAssistantId
      assistantConfig = assistantManager.getAssistantConfig(defaultAssistantId) ?? undefined
    }

    let skillCtx: SkillContext | undefined
    const resolvedChartAutoMode: ChartAutoMode = chartAutoMode ?? 'suggest'
    const chartRuntime = resolveChartRuntimeForRequest({
      skillId,
      userMessage,
      locale,
      assistantAllowedTools: assistantConfig?.allowedBuiltinTools,
      enableAutoDetection: enableAutoSkill === true,
      chartAutoMode: resolvedChartAutoMode,
    })
    if (chartRuntime.isChartCapability) {
      const chartSkill = chartRuntime.skillDef ?? getChartCapabilitySkill(locale ?? 'zh-CN')
      skillCtx = { skillDef: { ...chartSkill, chatScope: 'all' } as SkillContext['skillDef'] }
      assistantConfig = {
        ...(assistantConfig ?? {
          id: resolvedAssistantId,
          name: resolvedAssistantId,
          systemPrompt: '',
          presetQuestions: [],
        }),
        allowedBuiltinTools: chartRuntime.allowedBuiltinTools,
      }
    } else if (skillId) {
      const skillDef = skillManager.getSkillConfig(skillId) ?? undefined
      if (skillDef) {
        skillCtx = { skillDef }
      }
    } else if (enableAutoSkill) {
      const effectiveChatType = chatType ?? 'group'
      const autoSkillAllowedTools =
        resolvedChartAutoMode === 'explicit'
          ? (assistantConfig?.allowedBuiltinTools ?? [...CHART_CAPABILITY_ANALYSIS_TOOLS])
          : (getAllowedBuiltinToolsForChartAutoSkill(assistantConfig?.allowedBuiltinTools) ?? [
              ...CHART_CAPABILITY_ANALYSIS_TOOLS,
            ])
      assistantConfig = {
        ...(assistantConfig ?? {
          id: resolvedAssistantId,
          name: resolvedAssistantId,
          systemPrompt: '',
          presetQuestions: [],
        }),
        allowedBuiltinTools: autoSkillAllowedTools,
      }
      const allowedTools = autoSkillAllowedTools
      const baseMenu = skillManager.getSkillMenu(effectiveChatType, allowedTools)
      const menu =
        resolvedChartAutoMode === 'aggressive'
          ? buildSkillMenuWithBuiltinChart(baseMenu, locale, allowedTools)
          : baseMenu
      if (menu) {
        skillCtx = { skillMenu: menu }
      }
    }

    const maxToolResultPercent = compressionConfig?.maxToolResultPercent ?? 50
    const modelDef = findModelDefinition(activeAIConfig.provider, activeAIConfig.model || '')
    const resolvedContextWindow = modelDef?.contextWindow || DEFAULT_CONTEXT_WINDOW
    const maxToolResultTokens = Math.floor(resolvedContextWindow * (maxToolResultPercent / 100))

    let dataSnapshot: ToolContext['dataSnapshot'] | undefined
    try {
      dataSnapshot = createDataSnapshotFromOverview(await workerManager.getChatOverview(sessionId, 10))
    } catch (error) {
      aiLogger.warn('AgentStream', `Failed to load data snapshot: ${requestId}`, { error: String(error) })
    }

    const context: ToolContext = {
      sessionId,
      aiChatId,
      historyLeafMessageId: historyLeafMessageId ?? undefined,
      timeFilter: params.timeFilter,
      maxMessagesLimit: params.maxMessagesLimit,
      ownerInfo,
      mentionedMembers: mentionedMembers as ToolContext['mentionedMembers'],
      preprocessConfig: params.preprocessConfig as ToolContext['preprocessConfig'],
      maxToolResultTokens,
      dataSnapshot,
      abortSignal,
    }

    if (isCodexCliProvider(activeAIConfig.provider)) {
      const toolFallback = getCodexCliToolFallback({
        userMessage,
        requestedToolNames: skillCtx?.skillDef?.tools,
        locale,
      })
      if (toolFallback) {
        onEvent({ type: 'content', content: toolFallback })
        onEvent({
          type: 'done',
          isFinished: true,
          usage: {
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            cacheReadTokens: 0,
            cacheWriteTokens: 0,
          },
        })
        return
      }

      const contextTools = getAllTools(context, assistantConfig?.allowedBuiltinTools)
      const loadChatContext = (messageLimit: number | undefined) =>
        buildCodexCliChatContext({
          hasSelectedChat: Boolean(sessionId),
          tools: contextTools,
          dataSnapshot,
          maxMessagesLimit: messageLimit,
          locale,
          abortSignal,
        })

      let chatContext: string
      try {
        chatContext = await loadChatContext(params.maxMessagesLimit)
      } catch (error) {
        const message = formatCodexCliChatContextError(error, locale)
        const serializedError = serializeError(new Error(message), activeAIConfig.provider)
        serializedError.friendlyMessage = message
        aiLogger.error('AgentStream', `Codex CLI chat context error: ${requestId}`, serializedError)
        onEvent({ type: 'error', error: serializedError, isFinished: true })
        return
      }

      let history: Array<{ role: 'user' | 'assistant' | 'summary'; content: string }> = []
      const loadHistory = () => {
        try {
          return getHistoryForAgent(aiChatId, undefined, historyLeafMessageId)
        } catch {
          return []
        }
      }

      const providerSystemPrompt = buildSystemPrompt(
        chatType ?? 'group',
        assistantConfig?.systemPrompt,
        ownerInfo,
        locale ?? 'zh-CN',
        skillCtx,
        mentionedMembers as ToolContext['mentionedMembers'],
        dataSnapshot
      )
      let providerSystemPromptWithChatContext = `${providerSystemPrompt}\n\n${chatContext}`
      const requestCodex = () =>
        runCodexCli({
          messages: [
            { role: 'system', content: providerSystemPromptWithChatContext },
            ...history.map((message) => ({
              role: message.role === 'summary' ? 'assistant' : message.role,
              content: message.content,
            })),
            { role: 'user', content: userMessage },
          ],
          abortSignal,
        })

      history = loadHistory()

      try {
        let content: string
        try {
          content = await requestCodex()
        } catch (error) {
          if (
            !(error instanceof CodexCliError) ||
            error.code !== 'CONTEXT_TOO_LONG' ||
            historyLeafMessageId !== undefined
          ) {
            throw error
          }

          if (compressionConfig?.enabled) {
            const compressionResult = await manualCompress(
              aiChatId,
              compressionConfig as CompressionConfig,
              providerSystemPromptWithChatContext,
              buildCompressionAdapter(activeAIConfig, undefined, abortSignal),
              getAIChatManager(),
              compressionLogger
            )
            if (compressionResult.compressed && compressionResult.summaryContent) {
              onEvent({
                type: 'compression_done',
                compressionResult: {
                  summaryContent: compressionResult.summaryContent,
                  tokensBefore: compressionResult.tokensBefore ?? 0,
                  tokensAfter: compressionResult.tokensAfter ?? 0,
                  timestamp: Date.now(),
                },
              })
            }
          }
          chatContext = await loadChatContext(getCodexCliRetryMessageLimit(params.maxMessagesLimit))
          providerSystemPromptWithChatContext = `${providerSystemPrompt}\n\n${chatContext}`
          history = loadHistory()
          content = await requestCodex()
        }

        if (!abortSignal.aborted) onEvent({ type: 'content', content })
        onEvent({
          type: 'done',
          isFinished: true,
          usage: {
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            cacheReadTokens: 0,
            cacheWriteTokens: 0,
          },
        })
      } catch (error) {
        if (abortSignal.aborted) return
        const message = formatCodexCliError(error, locale)
        const serializedError = serializeError(new Error(message), activeAIConfig.provider)
        serializedError.friendlyMessage = message
        aiLogger.error('AgentStream', `Codex CLI execution error: ${requestId}`, serializedError)
        onEvent({ type: 'error', error: serializedError, isFinished: true })
      }
      return
    }

    const piModel = buildPiModel(activeAIConfig)
    const agent = new Agent(
      context,
      piModel,
      activeAIConfig.apiKey,
      {
        abortSignal,
        thinkingLevel: thinkingLevel as import('@openchatlab/core').ThinkingLevel | undefined,
        chartAutoMode: resolvedChartAutoMode,
      },
      chatType ?? 'group',
      locale ?? 'zh-CN',
      assistantConfig,
      skillCtx
    )

    try {
      await agent.executeStream(userMessage, (chunk: AgentStreamChunk) => {
        if (abortSignal.aborted) return
        onEvent(chunk as SharedAgentStreamChunk)
      })
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return
      const serializedError = serializeError(error, activeAIConfig.provider)
      serializedError.friendlyMessage = formatAIError(error, {
        providerName: resolveProviderName(activeAIConfig.provider),
        rawErrorLabel: t('llm.rawErrorLabel'),
      })
      if (!serializedError.url && activeAIConfig.baseUrl) serializedError.url = activeAIConfig.baseUrl
      aiLogger.error('AgentStream', `Agent execution error: ${requestId}`, serializedError)
      onEvent({ type: 'error', error: serializedError, isFinished: true })
    }
  }
}
