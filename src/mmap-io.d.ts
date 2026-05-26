declare module '@fayzanx/mmap-io' {
  const mod: {
    PROT_READ: number
    PROT_WRITE: number
    PROT_EXEC: number
    PROT_NONE: number
    MAP_SHARED: number
    MAP_PRIVATE: number
    MAP_ANONYMOUS: number
    map: (size: number, protection: number, flags: number, fd: number, offset?: number) => Buffer
    advise: (buffer: Buffer, advise: number) => void
    incore: (buffer: Buffer) => [number, number]
    sync: (buffer: Buffer) => void
  }
  export default mod
}
