import { constants as fsConstants } from 'node:fs'
import { access, mkdtemp, rm } from 'node:fs/promises'
import { homedir, tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { extractToolResultText } from '@openchatlab/core'
import type { CompressionLlmAdapter } from './compression/types'
import { shouldUseChartCapabilityForMessage } from './chart-runtime'

export const CODEX_CLI_PROVIDER_ID = 'codex-cli'
/** ChatLab-side budget only; the effective CLI/model/account limit may be lower. */
export const CODEX_CLI_CONTEXT_WINDOW = 1_000_000
export const DEFAULT_CODEX_CLI_TIMEOUT_MS = 120_000

const SENSITIVE_ENV_NAMES = new Set([
  'OPENAI_API_KEY',
  'CODEX_API_KEY',
  'CODEX_ACCESS_TOKEN',
  'OPENAI_ACCESS_TOKEN',
  'ANTHROPIC_API_KEY',
  'GEMINI_API_KEY',
  'DEEPSEEK_API_KEY',
  'MOONSHOT_API_KEY',
])
const SENSITIVE_ENV_NAME = /(?:^|_)(?:SECRET|PASSWORD|PRIVATE_KEY)(?:_|$)/i
const ANSI_ESCAPE = new RegExp(`${String.fromCharCode(27)}\\[[0-?]*[ -/]*[@-~]`, 'g')
const SECRET_ASSIGNMENT =
  /((?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|authorization|bearer)\s*(?::|=)\s*)("[^"]*"|'[^']*'|[^\s,;]+)/gi
const SECRET_PREFIX = /\b(?:sk|sess|rk)-[a-z0-9_-]{8,}\b/gi
const MAX_STDERR_SUMMARY_CHARS = 2000

export type CodexCliErrorCode =
  | 'NOT_INSTALLED'
  | 'NOT_LOGGED_IN'
  | 'UNTRUSTED_DIRECTORY'
  | 'UNSUPPORTED_SERVICE_TIER'
  | 'UNSUPPORTED_ARGUMENT'
  | 'CONTEXT_TOO_LONG'
  | 'EXECUTION_FAILED'
  | 'EMPTY_OUTPUT'
  | 'TIMEOUT'
  | 'ABORTED'

export class CodexCliError extends Error {
  readonly code: CodexCliErrorCode
  readonly stderrSummary?: string

  constructor(code: CodexCliErrorCode, stderrSummary?: string) {
    super(stderrSummary ?? code)
    this.name = 'CodexCliError'
    this.code = code
    this.stderrSummary = stderrSummary
  }
}

export interface CodexCliMessage {
  role: string
  content: string
}

export interface RunCodexCliOptions {
  messages?: CodexCliMessage[]
  prompt?: string
  timeoutMs?: number
  abortSignal?: AbortSignal
  commandPath?: string
}

export interface CreateCodexCliCompressionAdapterOptions {
  abortSignal?: AbortSignal
  onCompressing?: () => void
  onError?: (error: unknown) => void
}

export type CodexCliChatContextErrorCode = 'NO_CHAT_CONTEXT' | 'READ_CHAT_CONTEXT_FAILED'

export class CodexCliChatContextError extends Error {
  readonly code: CodexCliChatContextErrorCode

  constructor(code: CodexCliChatContextErrorCode, message?: string) {
    super(message ?? code)
    this.name = 'CodexCliChatContextError'
    this.code = code
  }
}

export interface CodexCliChatContextTool {
  name: string
  execute(toolCallId: string, params: unknown, signal?: AbortSignal): Promise<unknown>
}

export interface CodexCliChatSnapshot {
  name: string
  totalMessages: number
  firstMessageTs: number | null
  lastMessageTs: number | null
}

export interface BuildCodexCliChatContextOptions {
  hasSelectedChat: boolean
  tools: readonly CodexCliChatContextTool[]
  dataSnapshot?: CodexCliChatSnapshot
  maxMessagesLimit?: number
  locale?: string
  abortSignal?: AbortSignal
}

interface ProcessResult {
  stdout: string
  stderr: string
  exitCode: number | null
  spawnError?: NodeJS.ErrnoException
  timedOut: boolean
  aborted: boolean
}

export function isCodexCliProvider(provider: string | undefined): boolean {
  return provider === CODEX_CLI_PROVIDER_ID
}

export function filterCodexCliEnvironment(environment: NodeJS.ProcessEnv = process.env): Record<string, string> {
  const filtered: Record<string, string> = {}
  for (const [name, value] of Object.entries(environment)) {
    if (value === undefined || SENSITIVE_ENV_NAMES.has(name.toUpperCase()) || SENSITIVE_ENV_NAME.test(name)) continue
    filtered[name] = value
  }
  return filtered
}

export function sanitizeCodexCliStderr(stderr: string): string {
  return stderr
    .replace(ANSI_ESCAPE, '')
    .replace(SECRET_ASSIGNMENT, '$1[redacted]')
    .replace(SECRET_PREFIX, '[redacted]')
    .trim()
    .slice(0, MAX_STDERR_SUMMARY_CHARS)
}

export function buildCodexCliPrompt(messages: CodexCliMessage[]): string {
  const conversation = messages
    .filter((message) => typeof message.content === 'string' && message.content.trim().length > 0)
    .map((message) => `[${message.role || 'user'}]\n${message.content}`)
    .join('\n\n')

  return `You are being used as a model provider inside ChatLab.
Please answer the latest user message based on the conversation below.
Do not use tools, shell commands, files, or web search.
ChatLab internal function calling is not connected to this provider. Never claim that you queried the
ChatLab database, executed SQL, or generated a native ChatLab chart.
Return only the assistant response.

Conversation:
${conversation}`
}

export function getCodexCliToolFallback(options: {
  userMessage: string
  requestedToolNames?: readonly string[]
  locale?: string
}): string | null {
  const requestedTools = new Set(options.requestedToolNames ?? [])
  const requestsChart = requestedTools.has('render_chart') || shouldUseChartCapabilityForMessage(options.userMessage)
  const requestsDirectSql =
    requestedTools.has('execute_sql') &&
    /(?:执行|运行).{0,8}\bsql\b|\bsql\b.{0,8}(?:执行|运行)|\b(?:run|execute)\s+(?:an?\s+)?sql\b/i.test(
      options.userMessage
    )
  if (!requestsChart && !requestsDirectSql) return null

  const isZh = (options.locale ?? 'zh-CN').toLowerCase().startsWith('zh')
  if (requestsChart) {
    return isZh
      ? 'Codex CLI Provider 当前可以读取 ChatLab 提供的聊天上下文并生成图表分析建议，但尚未接入 ChatLab 原生图表渲染。请切换支持图表工具的 API Provider 以生成可视化图表。'
      : 'Codex CLI Provider can read ChatLab-provided chat context and suggest chart analysis, but native ChatLab chart rendering is not connected. Switch to an API provider with chart tools to generate a visual chart.'
  }
  return isZh
    ? 'Codex CLI Provider 不会直接执行 SQL。当前可以基于 ChatLab 安全提供的聊天记录上下文进行分析；如需执行自定义 SQL，请切换支持内部工具调用的 API Provider。'
    : 'Codex CLI Provider does not execute SQL directly. It can analyze chat context safely provided by ChatLab; switch to an API provider with internal tool calling to execute custom SQL.'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function normalizeMessageLimit(value: number | undefined): number {
  if (!Number.isFinite(value) || !value) return 100
  return Math.min(Math.max(Math.trunc(value), 1), 50_000)
}

export function getCodexCliRetryMessageLimit(value: number | undefined): number {
  const current = normalizeMessageLimit(value)
  return Math.max(100, Math.min(1000, Math.floor(current / 2)))
}

function formatSnapshotTime(timestamp: number | null | undefined, locale: string): string {
  if (!timestamp) return locale.startsWith('zh') ? '未知' : 'unknown'
  return new Date(timestamp * 1000).toLocaleString(locale.startsWith('zh') ? 'zh-CN' : 'en-US')
}

export async function buildCodexCliChatContext(options: BuildCodexCliChatContextOptions): Promise<string> {
  const locale = options.locale ?? 'zh-CN'
  const isZh = locale.toLowerCase().startsWith('zh')
  if (!options.hasSelectedChat) {
    throw new CodexCliChatContextError('NO_CHAT_CONTEXT')
  }

  const recentMessagesTool = options.tools.find((tool) => tool.name === 'get_recent_messages')
  if (!recentMessagesTool) {
    throw new CodexCliChatContextError('READ_CHAT_CONTEXT_FAILED', 'get_recent_messages tool is unavailable')
  }

  const maxMessagesLimit = normalizeMessageLimit(options.maxMessagesLimit)

  try {
    const result = await recentMessagesTool.execute(
      'codex-cli-chat-context',
      { limit: maxMessagesLimit },
      options.abortSignal
    )
    const text = extractToolResultText(result).trim()
    if (!text || /^error:/i.test(text)) {
      throw new Error(text || 'get_recent_messages returned no readable context')
    }

    const details = isRecord(result) && isRecord(result.details) ? result.details : {}
    const returned =
      typeof details.returned === 'number' && Number.isFinite(details.returned)
        ? details.returned
        : Math.min(options.dataSnapshot?.totalMessages ?? maxMessagesLimit, maxMessagesLimit)
    const total =
      typeof details.total === 'number' && Number.isFinite(details.total)
        ? details.total
        : (options.dataSnapshot?.totalMessages ?? returned)
    const name = options.dataSnapshot?.name || (isZh ? '当前选中聊天' : 'Current selected chat')
    const timeRange =
      details.timeRange ??
      `${formatSnapshotTime(options.dataSnapshot?.firstMessageTs, locale)} - ${formatSnapshotTime(
        options.dataSnapshot?.lastMessageTs,
        locale
      )}`

    if (isZh) {
      return `ChatLab Context:
当前聊天对象：${name}
消息总数：${total}
本次消息上限：${maxMessagesLimit}
本次实际发送消息数：${returned}
时间范围：${typeof timeRange === 'string' ? timeRange : JSON.stringify(timeRange)}
上下文来源：ChatLab 已导入聊天记录，通过 ChatLab 安全数据访问层读取

重要说明：
- ChatLab 已经在下方提供了当前可用的聊天记录，请直接基于这些数据回答。
- 不要声称你无法读取 ChatLab 数据库或聊天记录。
- 如果这些记录不足以支持完整结论，请说“当前传入的上下文不足”，并明确分析范围。
- “所有内容”受本次消息上限和上下文预算约束，不代表数据库中的全部 ${total} 条消息都已发送。

本次可用聊天记录：
${text}`
    }

    return `ChatLab Context:
Current chat: ${name}
Total messages: ${total}
Configured message limit: ${maxMessagesLimit}
Messages included in this request: ${returned}
Time range: ${typeof timeRange === 'string' ? timeRange : JSON.stringify(timeRange)}
Source: ChatLab imported chat records, read through ChatLab's safe data-access layer

Important:
- ChatLab has provided the available chat records below. Base your answer on this data.
- Do not claim that you cannot read the ChatLab database or chat records.
- If the records are insufficient, say "the context provided for this request is insufficient" and state the scope.
- "All content" is bounded by the configured message limit and context budget; not all ${total} database messages are necessarily included.

Available chat records:
${text}`
  } catch (error) {
    if (error instanceof CodexCliChatContextError) throw error
    const message = error instanceof Error ? error.message : String(error)
    throw new CodexCliChatContextError('READ_CHAT_CONTEXT_FAILED', message)
  }
}

export function formatCodexCliChatContextError(error: unknown, locale = 'zh-CN'): string {
  const isZh = locale.toLowerCase().startsWith('zh')
  if (error instanceof CodexCliChatContextError && error.code === 'NO_CHAT_CONTEXT') {
    return isZh
      ? '当前没有可用聊天记录上下文，请先选择或导入聊天记录。'
      : 'No chat context is currently available. Select or import a chat first.'
  }
  const detail = error instanceof Error && error.message ? error.message : isZh ? '未知错误' : 'unknown error'
  return isZh ? `读取 ChatLab 聊天上下文失败：${detail}` : `Failed to read ChatLab chat context: ${detail}`
}

function executableNames(platform: NodeJS.Platform): string[] {
  return platform === 'win32' ? ['codex.exe', 'codex.cmd', 'codex'] : ['codex']
}

function candidateDirectories(environment: NodeJS.ProcessEnv, home: string): string[] {
  const pathDirectories = (environment.PATH ?? '').split(delimiter).filter(Boolean)
  return Array.from(
    new Set([...pathDirectories, '/usr/local/bin', '/opt/homebrew/bin', join(home, '.local', 'bin'), join(home, 'bin')])
  )
}

export async function findCodexCliCommand(
  environment: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
  home: string = homedir()
): Promise<string | null> {
  for (const directory of candidateDirectories(environment, home)) {
    for (const executable of executableNames(platform)) {
      const candidate = join(directory, executable)
      try {
        await access(candidate, platform === 'win32' ? fsConstants.F_OK : fsConstants.X_OK)
        return candidate
      } catch {
        // Continue checking the remaining PATH and common installation locations.
      }
    }
  }
  return null
}

function terminateChild(child: ChildProcessWithoutNullStreams): void {
  if (child.exitCode !== null || child.signalCode !== null) return
  child.kill('SIGTERM')
  const forceKillTimer = setTimeout(() => {
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL')
  }, 2_000)
  forceKillTimer.unref()
}

function runProcess(
  command: string,
  args: string[],
  options: {
    cwd: string
    stdin?: string
    timeoutMs: number
    abortSignal?: AbortSignal
  }
): Promise<ProcessResult> {
  return new Promise((resolve) => {
    let stdout = ''
    let stderr = ''
    let timedOut = false
    let aborted = options.abortSignal?.aborted ?? false
    let settled = false

    const child = spawn(command, args, {
      cwd: options.cwd,
      env: filterCodexCliEnvironment(),
      shell: false,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    })

    const finish = (result: ProcessResult) => {
      if (settled) return
      settled = true
      clearTimeout(timeout)
      options.abortSignal?.removeEventListener('abort', abort)
      resolve(result)
    }

    const abort = () => {
      aborted = true
      terminateChild(child)
    }

    const timeout = setTimeout(() => {
      timedOut = true
      terminateChild(child)
    }, options.timeoutMs)
    timeout.unref()

    options.abortSignal?.addEventListener('abort', abort, { once: true })
    if (aborted) terminateChild(child)

    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => {
      stdout += chunk
    })
    child.stderr.on('data', (chunk: string) => {
      stderr += chunk
    })
    child.stdin.on('error', () => {
      // The exit code and stderr provide the useful failure classification.
    })

    child.once('error', (error: NodeJS.ErrnoException) => {
      finish({ stdout, stderr, exitCode: null, spawnError: error, timedOut, aborted })
    })
    child.once('close', (exitCode) => {
      finish({ stdout, stderr, exitCode, timedOut, aborted })
    })

    child.stdin.end(options.stdin ?? '')
  })
}

function classifyProcessFailure(result: ProcessResult): CodexCliError {
  if (result.aborted) return new CodexCliError('ABORTED')
  if (result.timedOut) return new CodexCliError('TIMEOUT')
  if (result.spawnError?.code === 'ENOENT') return new CodexCliError('NOT_INSTALLED')

  const diagnostic = `${result.stderr}\n${result.stdout}`.toLowerCase()
  const stderrSummary = sanitizeCodexCliStderr(result.stderr)
  if (diagnostic.includes('not inside a trusted directory')) {
    return new CodexCliError('UNTRUSTED_DIRECTORY', stderrSummary)
  }
  if (
    diagnostic.includes('unsupported service_tier') ||
    diagnostic.includes('unsupported service tier') ||
    (diagnostic.includes('service_tier') &&
      (diagnostic.includes('not supported') || diagnostic.includes('unavailable')))
  ) {
    return new CodexCliError('UNSUPPORTED_SERVICE_TIER', stderrSummary)
  }
  if (
    diagnostic.includes('unknown option') ||
    diagnostic.includes('unexpected argument') ||
    diagnostic.includes('unrecognized option')
  ) {
    return new CodexCliError('UNSUPPORTED_ARGUMENT', stderrSummary)
  }
  if (
    diagnostic.includes('context length exceeded') ||
    diagnostic.includes('maximum context length') ||
    diagnostic.includes('context window exceeded') ||
    diagnostic.includes('too many tokens') ||
    diagnostic.includes('input is too long') ||
    diagnostic.includes('request too large')
  ) {
    return new CodexCliError('CONTEXT_TOO_LONG', stderrSummary)
  }
  if (
    diagnostic.includes('not logged in') ||
    diagnostic.includes('login required') ||
    diagnostic.includes('please log in') ||
    diagnostic.includes('authentication required') ||
    diagnostic.includes('sign in with chatgpt') ||
    /\bauth(?:entication|orization)?\b/.test(diagnostic)
  ) {
    return new CodexCliError('NOT_LOGGED_IN', stderrSummary)
  }

  return new CodexCliError('EXECUTION_FAILED', stderrSummary)
}

async function resolveCommand(commandPath?: string): Promise<string> {
  if (commandPath) return commandPath
  const command = await findCodexCliCommand()
  if (!command) throw new CodexCliError('NOT_INSTALLED')
  return command
}

export async function validateCodexCli(
  options: {
    timeoutMs?: number
    abortSignal?: AbortSignal
    commandPath?: string
  } = {}
): Promise<void> {
  const command = await resolveCommand(options.commandPath)
  const workingDirectory = await mkdtemp(join(tmpdir(), 'chatlab-codex-'))

  try {
    const result = await runProcess(command, ['login', 'status'], {
      cwd: workingDirectory,
      timeoutMs: Math.min(options.timeoutMs ?? DEFAULT_CODEX_CLI_TIMEOUT_MS, 15_000),
      abortSignal: options.abortSignal,
    })

    if (result.spawnError || result.exitCode !== 0 || result.timedOut || result.aborted) {
      const classified = classifyProcessFailure(result)
      if (classified.code === 'EXECUTION_FAILED') {
        throw new CodexCliError('NOT_LOGGED_IN', classified.stderrSummary)
      }
      throw classified
    }
  } finally {
    await rm(workingDirectory, { recursive: true, force: true })
  }
}

export async function runCodexCli(options: RunCodexCliOptions): Promise<string> {
  const command = await resolveCommand(options.commandPath)
  const timeoutMs = options.timeoutMs ?? DEFAULT_CODEX_CLI_TIMEOUT_MS
  const prompt = options.prompt ?? buildCodexCliPrompt(options.messages ?? [])

  await validateCodexCli({
    commandPath: command,
    timeoutMs,
    abortSignal: options.abortSignal,
  })

  const workingDirectory = await mkdtemp(join(tmpdir(), 'chatlab-codex-'))
  const args = ['exec', '--skip-git-repo-check', '-']

  try {
    const result = await runProcess(command, args, {
      cwd: workingDirectory,
      stdin: prompt,
      timeoutMs,
      abortSignal: options.abortSignal,
    })

    if (result.spawnError || result.exitCode !== 0 || result.timedOut || result.aborted) {
      throw classifyProcessFailure(result)
    }

    const output = result.stdout.trim()
    if (!output) throw new CodexCliError('EMPTY_OUTPUT')
    return output
  } finally {
    await rm(workingDirectory, { recursive: true, force: true })
  }
}

export function createCodexCliCompressionAdapter(
  options: CreateCodexCliCompressionAdapterOptions = {}
): CompressionLlmAdapter {
  return {
    contextWindow: CODEX_CLI_CONTEXT_WINDOW,
    async compress(prompt: string): Promise<string | null> {
      options.onCompressing?.()
      try {
        return await runCodexCli({ prompt, abortSignal: options.abortSignal })
      } catch (error) {
        options.onError?.(error)
        return null
      }
    },
  }
}

const ERROR_MESSAGES: Record<CodexCliErrorCode, { zh: string; en: string }> = {
  NOT_INSTALLED: {
    zh: '未检测到 Codex CLI，请先安装 OpenAI Codex CLI。',
    en: 'Codex CLI was not detected. Install OpenAI Codex CLI first.',
  },
  NOT_LOGGED_IN: {
    zh: 'Codex CLI 可能尚未登录，请先在终端运行 codex 并选择 Sign in with ChatGPT。',
    en: 'Codex CLI may not be signed in. Run codex in a terminal and choose Sign in with ChatGPT.',
  },
  UNTRUSTED_DIRECTORY: {
    zh: 'Codex CLI 报告当前目录不受信任。执行器已传入 --skip-git-repo-check，请确认当前 Codex CLI 版本支持该参数。',
    en: 'Codex CLI reported that the directory is not trusted. The executor already passes --skip-git-repo-check; verify that the installed Codex CLI supports this option.',
  },
  UNSUPPORTED_SERVICE_TIER: {
    zh: 'Codex CLI 配置中的 service_tier 当前不可用，请检查 ~/.codex/config.toml，尝试删除 service_tier 或改为支持的值。',
    en: 'The service_tier in the Codex CLI configuration is unavailable. Check ~/.codex/config.toml and remove it or change it to a supported value.',
  },
  UNSUPPORTED_ARGUMENT: {
    zh: 'Codex CLI 不支持当前调用参数，请检查下方错误并升级 Codex CLI 或调整参数。',
    en: 'Codex CLI does not support one of the invocation arguments. Review the error below and upgrade Codex CLI or adjust the argument.',
  },
  CONTEXT_TOO_LONG: {
    zh: 'Codex CLI 当前模型或账号不接受这么长的上下文，请减少发送条数、降低上下文窗口或开启摘要。',
    en: 'The current Codex CLI model or account does not accept this much context. Send fewer messages, lower the context window, or enable summaries.',
  },
  EXECUTION_FAILED: {
    zh: 'Codex CLI 执行失败。',
    en: 'Codex CLI execution failed.',
  },
  EMPTY_OUTPUT: {
    zh: 'Codex CLI 输出为空。',
    en: 'Codex CLI returned an empty response.',
  },
  TIMEOUT: {
    zh: 'Codex CLI 请求超时。',
    en: 'The Codex CLI request timed out.',
  },
  ABORTED: {
    zh: '用户已取消请求。',
    en: 'The request was cancelled.',
  },
}

export function formatCodexCliError(error: unknown, locale = 'zh-CN'): string {
  const code = error instanceof CodexCliError ? error.code : 'EXECUTION_FAILED'
  const baseMessage = locale.toLowerCase().startsWith('zh') ? ERROR_MESSAGES[code].zh : ERROR_MESSAGES[code].en
  const stderrSummary = error instanceof CodexCliError ? error.stderrSummary : undefined
  if (!stderrSummary) return baseMessage
  return locale.toLowerCase().startsWith('zh')
    ? `${baseMessage.replace(/。$/, '')}：${stderrSummary}`
    : `${baseMessage.replace(/\.$/, '')}: ${stderrSummary}`
}
