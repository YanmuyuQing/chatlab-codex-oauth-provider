import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import vm from 'node:vm'

import { patchMain, readArchive, sha256, stageRepair, verifyArchive } from './stage-terra-0376.mjs'

// The released detector does not recognize Terra. Exercise the patched function,
// not a source string assertion: that mistake adds an unsupported topic temperature.
const MAIN = Buffer.from(`const VM={gpt5:{low:"low",medium:"medium",high:"high"}};
function CE(e,t){const r=t.toLowerCase(),n=e.toLowerCase();if(n==="openai"||n==="openai-compatible"){if(/gpt-5/.test(r))return"gpt5";}return"none";}
function topicOptions(model){return CE("openai-compatible",model)==="none"?{temperature:.1}:{};}`)

const fixtures = []
test.afterEach(() => {
  for (const directory of fixtures.splice(0)) rmSync(directory, { recursive: true, force: true })
})

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'chatlab-terra-repair-'))
  fixtures.push(root)
  const appPath = join(root, 'ChatLab.app')
  mkdirSync(join(appPath, 'Contents', 'Resources'), { recursive: true })
  const before = Buffer.from('entry before main')
  const after = Buffer.from('entry after main')
  const entry = (bytes, offset) => ({
    size: bytes.length,
    offset: String(offset),
    integrity: { algorithm: 'SHA256', hash: sha256(bytes), blockSize: 4194304, blocks: [sha256(bytes)] },
  })
  const header = {
    files: {
      'before.txt': entry(before, 0),
      out: { files: { main: { files: { 'index.js': entry(MAIN, before.length) } } } },
      'after.txt': entry(after, before.length + MAIN.length),
      'empty.txt': entry(Buffer.alloc(0), before.length + MAIN.length + after.length),
      'unpacked.node': { size: 100, unpacked: true },
    },
  }
  const rawHeader = Buffer.from(JSON.stringify(header))
  const paddedSize = Math.ceil(rawHeader.length / 4) * 4
  const prefix = Buffer.alloc(16 + paddedSize)
  prefix.writeUInt32LE(4, 0)
  prefix.writeUInt32LE(8 + paddedSize, 4)
  prefix.writeUInt32LE(4 + paddedSize, 8)
  prefix.writeUInt32LE(rawHeader.length, 12)
  rawHeader.copy(prefix, 16)
  const archivePath = join(appPath, 'Contents', 'Resources', 'app.asar')
  writeFileSync(archivePath, Buffer.concat([prefix, before, MAIN, after]))
  const plistPath = join(appPath, 'Contents', 'Info.plist')
  writeFileSync(
    plistPath,
    `<plist><dict><key>CFBundleShortVersionString</key><string>0.37.6</string>
    <key>ElectronAsarIntegrity</key><dict><key>Resources/app.asar</key><dict>
    <key>algorithm</key><string>SHA256</string><key>hash</key><string>${sha256(rawHeader)}</string>
    </dict></dict></dict></plist>`
  )
  return { root, appPath, archivePath, plistPath, before, after }
}

test('staged detector recognizes Terra and omits unsupported topic sampling', () => {
  const baseline = vm.runInNewContext(`${MAIN};JSON.stringify(topicOptions('gpt-6.1-terra'))`)
  assert.equal(baseline, '{"temperature":0.1}')
  const patched = patchMain(MAIN)
  const output = JSON.parse(
    vm.runInNewContext(`${patched};JSON.stringify({
      terra:topicOptions('gpt-6.1-terra'),normal:topicOptions('gpt-4.1'),gpt5:topicOptions('gpt-5'),
      levels:VM.gpt6_1_terra,other:CE('other-provider','gpt-6.1-terra')})`)
  )
  assert.deepEqual(output.terra, {})
  assert.deepEqual(output.gpt5, {})
  assert.deepEqual(output.normal, { temperature: 0.1 })
  assert.equal(output.levels.off, 'none')
  assert.equal(output.levels.minimal, null)
  assert.equal(output.levels.low, 'low')
  assert.equal(output.other, 'none')
  assert.throws(() => patchMain(patched), /already applied/)
})

test('staging preserves neighboring payloads, repairs offsets and verifies all integrity hashes', () => {
  const fixtureData = fixture()
  const { root, appPath, archivePath, plistPath, before, after } = fixtureData
  const originalArchive = readFileSync(archivePath)
  const originalPlist = readFileSync(plistPath)
  const outputDir = join(root, 'staged')
  const result = stageRepair({ appPath, outputDir, expectedMainHash: sha256(MAIN) })
  const archive = verifyArchive(result.archivePath)
  const bytes = readFileSync(result.archivePath)
  assert.deepEqual(bytes.subarray(archive.payloadStart, archive.payloadStart + before.length), before)
  assert.deepEqual(bytes.subarray(archive.mainStart + archive.main.size), after)
  assert.equal(sha256(archive.rawHeader), result.headerSha256)
  assert.equal(sha256(archive.mainBytes), result.mainSha256)
  assert.match(readFileSync(result.plistPath, 'utf8'), new RegExp(result.headerSha256))
  assert.deepEqual(readFileSync(archivePath), originalArchive)
  assert.deepEqual(readFileSync(plistPath), originalPlist)
  assert.throws(() => stageRepair({ appPath, outputDir, expectedMainHash: sha256(MAIN) }), /EEXIST/)

  const check = stageRepair({ appPath, checkOnly: true, expectedMainHash: sha256(MAIN) })
  assert.equal(check.mainSha256, result.mainSha256)
  const corrupted = Buffer.from(bytes)
  corrupted[archive.mainStart] ^= 1
  writeFileSync(result.archivePath, corrupted)
  assert.throws(() => verifyArchive(result.archivePath), /integrity mismatch/)
  assert.throws(() => stageRepair({ appPath, checkOnly: true }), /Unrecognized/)
  assert.equal(readArchive(archivePath).main.size, MAIN.length)
})
