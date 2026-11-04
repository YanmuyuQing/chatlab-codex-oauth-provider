#!/usr/bin/env node
/** Stage the GPT-6.1 Terra capability repair for an existing ChatLab 0.37.6 app.
 * The installed application is never edited. A copied app must be signed again
 * after installing these staged resources; signing is deliberately separate.
 */

import { createHash } from 'node:crypto'
import { closeSync, mkdirSync, openSync, readFileSync, readSync, statSync, writeFileSync, writeSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const ORIGINAL_MAIN_SHA256 = 'ea726fefe05226e574bab86cacabadccd81ba49abbf95af43bbfa9d4add78588'
const MAIN_PATH = 'out/main/index.js'
const FAMILY_MARKER = 'const VM={gpt5:'
const PROVIDER_MARKER = 'if(n==="openai"||n==="openai-compatible"){'
const TERRA_MAP =
  'gpt6_1_terra:{off:"none",minimal:null,low:"low",medium:"medium",high:"high",xhigh:"xhigh",auto:null},'
const TERRA_DETECTION = 'if(r==="gpt-6.1-terra")return"gpt6_1_terra";'
const INTEGRITY_PLIST =
  /(<key>ElectronAsarIntegrity<\/key>\s*<dict>\s*<key>Resources\/app\.asar<\/key>\s*<dict>\s*<key>algorithm<\/key>\s*<string>SHA256<\/string>\s*<key>hash<\/key>\s*<string>)([a-f0-9]{64})(<\/string>)/

export function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex')
}

function readBytes(fd, position, size) {
  const buffer = Buffer.alloc(size)
  if (readSync(fd, buffer, 0, size, position) !== size) throw new Error('Truncated ASAR data')
  return buffer
}

function entries(header, prefix = '') {
  return Object.entries(header.files).flatMap(([name, entry]) => {
    const path = prefix ? `${prefix}/${name}` : name
    return entry.files ? entries(entry, path) : [{ path, entry }]
  })
}

export function readArchive(archivePath) {
  const fd = openSync(archivePath, 'r')
  try {
    const prefix = readBytes(fd, 0, 16)
    const headerSize = prefix.readUInt32LE(4)
    const pickleSize = prefix.readUInt32LE(8)
    const jsonSize = prefix.readUInt32LE(12)
    if (prefix.readUInt32LE(0) !== 4 || headerSize !== pickleSize + 4 || jsonSize > pickleSize - 4) {
      throw new Error('Unsupported ASAR header layout')
    }
    const rawHeader = readBytes(fd, 16, jsonSize)
    const header = JSON.parse(rawHeader.toString('utf8'))
    const files = entries(header)
    const main = files.find(({ path }) => path === MAIN_PATH)?.entry
    if (!main || main.unpacked || main.link) throw new Error('Expected a packed main entry')
    const payloadStart = 8 + headerSize
    const mainStart = payloadStart + Number(main.offset)
    return { header, rawHeader, files, main, payloadStart, mainStart, mainBytes: readBytes(fd, mainStart, main.size) }
  } finally {
    closeSync(fd)
  }
}

/** Verify packed payloads and every recorded block hash, without loading the archive in memory. */
export function verifyArchive(archivePath) {
  const archive = readArchive(archivePath)
  const fd = openSync(archivePath, 'r')
  try {
    for (const { path, entry } of archive.files) {
      if (entry.unpacked || entry.link || !entry.integrity) continue
      const integrity = entry.integrity
      if (integrity.algorithm !== 'SHA256' || !(integrity.blockSize > 0)) {
        throw new Error(`Unsupported integrity metadata: ${path}`)
      }
      const hash = createHash('sha256')
      const blocks = []
      for (let offset = 0; offset < entry.size; offset += integrity.blockSize) {
        const bytes = readBytes(
          fd,
          archive.payloadStart + Number(entry.offset) + offset,
          Math.min(integrity.blockSize, entry.size - offset)
        )
        hash.update(bytes)
        blocks.push(sha256(bytes))
      }
      if (entry.size === 0) blocks.push(sha256(Buffer.alloc(0)))
      if (hash.digest('hex') !== integrity.hash || JSON.stringify(blocks) !== JSON.stringify(integrity.blocks)) {
        throw new Error(`ASAR integrity mismatch: ${path}`)
      }
    }
  } finally {
    closeSync(fd)
  }
  return archive
}

export function patchMain(mainBytes) {
  const source = mainBytes.toString('utf8')
  if (source.includes(TERRA_DETECTION) || source.includes(TERRA_MAP)) throw new Error('Terra repair is already applied')
  for (const marker of [FAMILY_MARKER, PROVIDER_MARKER]) {
    if (source.split(marker).length !== 2) throw new Error('Expected ChatLab 0.37.6 capability code exactly once')
  }
  return Buffer.from(
    source
      .replace(FAMILY_MARKER, `const VM={${TERRA_MAP}gpt5:`)
      .replace(PROVIDER_MARKER, `${PROVIDER_MARKER}${TERRA_DETECTION}`)
  )
}

function makeHeader(header) {
  const json = Buffer.from(JSON.stringify(header))
  const paddedSize = Math.ceil(json.length / 4) * 4
  const bytes = Buffer.alloc(16 + paddedSize)
  bytes.writeUInt32LE(4, 0)
  bytes.writeUInt32LE(8 + paddedSize, 4)
  bytes.writeUInt32LE(4 + paddedSize, 8)
  bytes.writeUInt32LE(json.length, 12)
  json.copy(bytes, 16)
  return { bytes, json }
}

function copyRange(sourceFd, targetFd, position, size) {
  const chunkSize = 1024 * 1024
  for (let offset = 0; offset < size; offset += chunkSize) {
    const bytes = readBytes(sourceFd, position + offset, Math.min(chunkSize, size - offset))
    writeSync(targetFd, bytes)
  }
}

/** expectedMainHash is injectable only for synthetic archive tests; the CLI pins the official bundle. */
export function stageRepair({ appPath, outputDir, checkOnly = false, expectedMainHash = ORIGINAL_MAIN_SHA256 }) {
  const archivePath = join(appPath, 'Contents', 'Resources', 'app.asar')
  const plistPath = join(appPath, 'Contents', 'Info.plist')
  const archive = verifyArchive(archivePath)
  const plist = readFileSync(plistPath, 'utf8')
  if (!/<key>CFBundleShortVersionString<\/key>\s*<string>0\.37\.6<\/string>/.test(plist)) {
    throw new Error('This repair supports ChatLab 0.37.6 only')
  }
  const oldHeaderHash = sha256(archive.rawHeader)
  const plistMatch = INTEGRITY_PLIST.exec(plist)
  if (!plistMatch || plistMatch[2] !== oldHeaderHash) throw new Error('Info.plist ASAR header integrity mismatch')
  const mainBytes = patchMain(archive.mainBytes)
  if (sha256(archive.mainBytes) !== expectedMainHash) throw new Error('Unrecognized ChatLab 0.37.6 main bundle')
  const delta = mainBytes.length - archive.main.size
  const oldMainEnd = archive.mainStart + archive.main.size
  for (const { entry } of archive.files) {
    if (!entry.unpacked && !entry.link && Number(entry.offset) > Number(archive.main.offset)) {
      entry.offset = String(Number(entry.offset) + delta)
    }
  }
  archive.main.size = mainBytes.length
  archive.main.integrity.hash = sha256(mainBytes)
  archive.main.integrity.blocks = []
  for (let offset = 0; offset < mainBytes.length; offset += archive.main.integrity.blockSize) {
    archive.main.integrity.blocks.push(sha256(mainBytes.subarray(offset, offset + archive.main.integrity.blockSize)))
  }
  const header = makeHeader(archive.header)
  const headerHash = sha256(header.json)
  const repairedPlist = plist.replace(INTEGRITY_PLIST, (_, before, _hash, after) => `${before}${headerHash}${after}`)
  const result = {
    mainSha256: sha256(mainBytes),
    headerSha256: headerHash,
    checkedPackedEntries: archive.files.filter(({ entry }) => !entry.unpacked && !entry.link && entry.integrity).length,
  }
  if (checkOnly) return result
  if (!outputDir) throw new Error('An output directory is required')
  mkdirSync(outputDir)
  const outputArchive = join(outputDir, 'app.asar')
  const outputPlist = join(outputDir, 'Info.plist')
  const sourceFd = openSync(archivePath, 'r')
  const targetFd = openSync(outputArchive, 'wx')
  try {
    writeSync(targetFd, header.bytes)
    copyRange(sourceFd, targetFd, archive.payloadStart, archive.mainStart - archive.payloadStart)
    writeSync(targetFd, mainBytes)
    copyRange(sourceFd, targetFd, oldMainEnd, statSync(archivePath).size - oldMainEnd)
  } finally {
    closeSync(sourceFd)
    closeSync(targetFd)
  }
  writeFileSync(outputPlist, repairedPlist, { flag: 'wx' })
  const verified = verifyArchive(outputArchive)
  if (sha256(verified.rawHeader) !== headerHash || !verified.mainBytes.equals(mainBytes)) {
    throw new Error('Staged repair verification failed')
  }
  return { ...result, archivePath: outputArchive, plistPath: outputPlist }
}

function main() {
  const args = process.argv.slice(2)
  const value = (name) => args[args.indexOf(name) + 1]
  const appPath = args.includes('--app') ? value('--app') : '/Applications/ChatLab.app'
  const checkOnly = args.includes('--check')
  const outputDir = args.includes('--output-dir') ? value('--output-dir') : undefined
  if (!checkOnly && !outputDir) throw new Error('Usage: --check, or --output-dir <fresh-directory> [--app <app>]')
  console.log(JSON.stringify(stageRepair({ appPath, outputDir, checkOnly }), null, 2))
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main()
