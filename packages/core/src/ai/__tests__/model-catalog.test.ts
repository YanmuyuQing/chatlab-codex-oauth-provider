import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { getBuiltinModelById } from '../model-catalog'

describe('model catalog provider isolation', () => {
  it('gives Codex CLI a large ChatLab-side budget without claiming tool or vision support', () => {
    const codexCli = getBuiltinModelById('codex-cli', 'codex-cli')

    assert.ok(codexCli)
    assert.equal(codexCli.contextWindow, 1_000_000)
    assert.deepEqual(codexCli.capabilities, ['chat', 'reasoning'])
  })

  it('does not change another provider model capability or context configuration', () => {
    const gpt4o = getBuiltinModelById('openai', 'gpt-4o')

    assert.ok(gpt4o)
    assert.equal(gpt4o.contextWindow, 128_000)
    assert.deepEqual(gpt4o.capabilities, ['chat', 'vision', 'function_calling'])
  })
})
