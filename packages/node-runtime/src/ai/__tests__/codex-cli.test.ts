import assert from 'node:assert/strict'
import { chmod, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, it } from 'node:test'
import {
  CODEX_CLI_CONTEXT_WINDOW,
  CodexCliError,
  createCodexCliCompressionAdapter,
  filterCodexCliEnvironment,
  formatCodexCliError,
  getCodexCliToolFallback,
  runCodexCli,
  sanitizeCodexCliStderr,
} from '../codex-cli'

const temporaryDirectories: string[] = []

async function createFakeCodex(source: string): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'chatlab-fake-codex-'))
  temporaryDirectories.push(directory)
  const executable = join(directory, 'codex')
  await writeFile(executable, `#!${process.execPath}\n${source}`, 'utf8')
  await chmod(executable, 0o755)
  return executable
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })))
})

describe('Codex CLI executor', () => {
  it('uses the minimal exec arguments, stdin, and the user environment without leaking API credentials', async () => {
    process.env.OPENAI_API_KEY = 'must-not-leak'
    process.env.CHATLAB_TEST_SAFE_VALUE = 'visible'
    process.env.XDG_CONFIG_HOME = '/tmp/chatlab-xdg-config'

    const commandPath = await createFakeCodex(`
if (process.argv[2] === 'login') process.exit(0)
let input = ''
process.stdin.setEncoding('utf8')
process.stdin.on('data', (chunk) => { input += chunk })
process.stdin.on('end', () => {
  process.stdout.write(JSON.stringify({
    args: process.argv.slice(2),
    input,
    apiKey: process.env.OPENAI_API_KEY,
    safeValue: process.env.CHATLAB_TEST_SAFE_VALUE,
    home: process.env.HOME,
    path: process.env.PATH,
    xdgConfigHome: process.env.XDG_CONFIG_HOME
  }))
})
`)

    try {
      const output = await runCodexCli({
        commandPath,
        prompt: 'hello; $(touch should-not-run)',
      })
      const result = JSON.parse(output) as {
        args: string[]
        input: string
        apiKey?: string
        safeValue?: string
        home?: string
        path?: string
        xdgConfigHome?: string
      }

      assert.equal(result.input, 'hello; $(touch should-not-run)')
      assert.deepEqual(result.args, ['exec', '--skip-git-repo-check', '-'])
      assert.equal(result.args.includes('hello; $(touch should-not-run)'), false)
      assert.equal(result.apiKey, undefined)
      assert.equal(result.safeValue, 'visible')
      assert.equal(result.home, process.env.HOME)
      assert.equal(result.path, process.env.PATH)
      assert.equal(result.xdgConfigHome, '/tmp/chatlab-xdg-config')
    } finally {
      delete process.env.OPENAI_API_KEY
      delete process.env.CHATLAB_TEST_SAFE_VALUE
      delete process.env.XDG_CONFIG_HOME
    }
  })

  it('classifies a missing command', async () => {
    await assert.rejects(
      runCodexCli({ commandPath: join(tmpdir(), 'missing-chatlab-codex'), prompt: 'hello' }),
      (error: unknown) => error instanceof CodexCliError && error.code === 'NOT_INSTALLED'
    )
  })

  it('classifies a non-zero execution exit', async () => {
    const commandPath = await createFakeCodex(`
if (process.argv[2] === 'login') process.exit(0)
process.stderr.write('request failed; api_key=sk-super-secret-value; retry later')
process.exit(7)
`)

    await assert.rejects(
      runCodexCli({ commandPath, prompt: 'hello' }),
      (error: unknown) =>
        error instanceof CodexCliError &&
        error.code === 'EXECUTION_FAILED' &&
        formatCodexCliError(error).includes('retry later') &&
        formatCodexCliError(error).includes('[redacted]') &&
        !formatCodexCliError(error).includes('super-secret')
    )
  })

  it('recognizes common Codex configuration and argument errors', async () => {
    const cases: Array<{ stderr: string; code: CodexCliError['code']; message: RegExp }> = [
      {
        stderr: 'Unsupported service_tier: priority',
        code: 'UNSUPPORTED_SERVICE_TIER',
        message: /~\/\.codex\/config\.toml/,
      },
      {
        stderr: 'Not inside a trusted directory',
        code: 'UNTRUSTED_DIRECTORY',
        message: /--skip-git-repo-check/,
      },
      {
        stderr: "error: unexpected argument '--legacy-flag'",
        code: 'UNSUPPORTED_ARGUMENT',
        message: /--legacy-flag/,
      },
      {
        stderr: 'Authentication required: not logged in',
        code: 'NOT_LOGGED_IN',
        message: /Sign in with ChatGPT/,
      },
      {
        stderr: 'maximum context length exceeded',
        code: 'CONTEXT_TOO_LONG',
        message: /开启摘要/,
      },
    ]

    for (const testCase of cases) {
      const commandPath = await createFakeCodex(`
if (process.argv[2] === 'login') process.exit(0)
process.stderr.write(${JSON.stringify(testCase.stderr)})
process.exit(1)
`)

      await assert.rejects(runCodexCli({ commandPath, prompt: 'hello' }), (error: unknown) => {
        return (
          error instanceof CodexCliError &&
          error.code === testCase.code &&
          testCase.message.test(formatCodexCliError(error))
        )
      })
    }
  })

  it('terminates a timed-out request', async () => {
    const commandPath = await createFakeCodex(`
if (process.argv[2] === 'login') process.exit(0)
setInterval(() => {}, 1_000)
`)

    await assert.rejects(
      runCodexCli({ commandPath, prompt: 'hello', timeoutMs: 50 }),
      (error: unknown) => error instanceof CodexCliError && error.code === 'TIMEOUT'
    )
  })

  it('filters common secret-like environment variable names', () => {
    const filtered = filterCodexCliEnvironment({
      PATH: '/usr/bin',
      HOME: '/tmp/home',
      TMPDIR: '/tmp',
      USER: 'chatlab',
      SSH_AUTH_SOCK: '/tmp/ssh-agent',
      OPENAI_API_KEY: 'secret',
      CODEX_API_KEY: 'secret',
      CODEX_ACCESS_TOKEN: 'secret',
      DATABASE_PASSWORD: 'secret',
      APP_PRIVATE_KEY: 'secret',
    })

    assert.deepEqual(filtered, {
      PATH: '/usr/bin',
      HOME: '/tmp/home',
      TMPDIR: '/tmp',
      USER: 'chatlab',
      SSH_AUTH_SOCK: '/tmp/ssh-agent',
    })
  })

  it('returns an explicit fallback instead of silently ignoring ChatLab tool requests', () => {
    assert.match(
      getCodexCliToolFallback({
        userMessage: '请生成最近一个月的趋势图',
      }) ?? '',
      /暂不支持 ChatLab 内部工具调用/
    )
    assert.match(
      getCodexCliToolFallback({ userMessage: '请执行 SQL 查询聊天记录' }) ?? '',
      /切换到支持 function calling 的 API Provider/
    )
    assert.equal(getCodexCliToolFallback({ userMessage: '请解释这段 SQL 的含义' }), null)
  })

  it('uses the Codex ChatLab-side budget for the existing compression pipeline', () => {
    const adapter = createCodexCliCompressionAdapter()

    assert.equal(adapter.contextWindow, CODEX_CLI_CONTEXT_WINDOW)
    assert.equal(adapter.contextWindow, 1_000_000)
  })

  it('limits and sanitizes stderr summaries', () => {
    const summary = sanitizeCodexCliStderr(`\u001b[31merror\u001b[0m bearer=secret ${'x'.repeat(3000)}`)

    assert.equal(summary.includes('\u001b'), false)
    assert.equal(summary.includes('bearer=secret'), false)
    assert.equal(summary.includes('bearer=[redacted]'), true)
    assert.equal(summary.length, 2000)
  })
})
