/*************************************************************************
 * rdmaroute_net.h - self-contained NCCL v9 net plugin local header.
 *
 * We deliberately do NOT include nccl/net.h (it drags in 11 version
 * headers + common.h). Instead we reproduce exactly the minimal set of
 * declarations that nccl/net_v9.h needs, in the exact field order from the
 * upstream NCCL 2.30.4 headers.
 *************************************************************************/
#ifndef RDMAROUTE_NET_ABI_H_
#define RDMAROUTE_NET_ABI_H_

#include <stdint.h>
#include <stdlib.h>

/* Parse a (possibly signed) base-10 int WITHOUT libc atoi/strtol: newer libc
 * headers redirect those to __isoc23_strtol@GLIBC_2.38, which the runtime
 * container (glibc < 2.38) can't resolve -> dlopen fails. This keeps the .so's
 * glibc floor low enough to load anywhere on the fleet. */
static inline int rr_atoi(const char* s) {
  if (!s) return 0;
  while (*s == ' ' || *s == '\t') s++;
  int neg = 0;
  if (*s == '+' || *s == '-') { neg = (*s == '-'); s++; }
  int v = 0;
  while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
  return neg ? -v : v;
}

/* ---- from net.h ---- */
#define NCCL_NET_HANDLE_MAXSIZE 128
#define NCCL_PTR_HOST   0x1
#define NCCL_PTR_CUDA   0x2
#define NCCL_PTR_DMABUF 0x4
#define NCCL_NET_MAX_REQUESTS 32

/* net_v9.h references NCCL_NET_MAX_DEVS_PER_NIC_V11 inside
 * ncclNetVDeviceProps_v9_t. Upstream value is 4. */
#ifndef NCCL_NET_MAX_DEVS_PER_NIC_V11
#define NCCL_NET_MAX_DEVS_PER_NIC_V11 4
#endif

/* ---- from err.h (exact enum) ---- */
typedef enum { ncclSuccess            =  0,
               ncclUnhandledCudaError =  1,
               ncclSystemError        =  2,
               ncclInternalError      =  3,
               ncclInvalidArgument    =  4,
               ncclInvalidUsage       =  5,
               ncclRemoteError        =  6 } ncclResult_t;

/* ---- debug logger typedef (matches NCCL's ncclDebugLogger_t) ---- */
typedef enum {
  NCCL_LOG_NONE  = 0, NCCL_LOG_VERSION = 1, NCCL_LOG_WARN = 2,
  NCCL_LOG_INFO  = 3, NCCL_LOG_ABORT   = 4, NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags,
                                  const char* file, int line, const char* fmt, ...);

/* ---- from net_device.h (exact field order) ---- */
#define NCCL_NET_DEVICE_INVALID_VERSION 0x0
typedef enum {
  NCCL_NET_DEVICE_HOST      = 0,
  NCCL_NET_DEVICE_UNPACK    = 1,
  NCCL_NET_DEVICE_GIN_PROXY = 2,
  NCCL_NET_DEVICE_GIN_GDAKI = 3,
} ncclNetDeviceType;

typedef struct {
  ncclNetDeviceType netDeviceType;
  int    netDeviceVersion;
  void*  handle;
  size_t size;
  int    needsProxyProgress;
} ncclNetDeviceHandle_v7_t;
typedef ncclNetDeviceHandle_v7_t ncclNetDeviceHandle_v9_t;

/* ---- from net_v9.h (exact field order) ---- */
typedef struct {
  int ndevs;
  int devs[NCCL_NET_MAX_DEVS_PER_NIC_V11];
} ncclNetVDeviceProps_v9_t;

typedef struct {
  char*    name;
  char*    pciPath;
  uint64_t guid;
  int      ptrSupport;
  int      regIsGlobal;
  int      forceFlush;
  int      speed;
  int      port;
  float    latency;
  int      maxComms;
  int      maxRecvs;
  ncclNetDeviceType netDeviceType;
  int      netDeviceVersion;
  ncclNetVDeviceProps_v9_t vProps;
  size_t   maxP2pBytes;
  size_t   maxCollBytes;
} ncclNetProperties_v9_t;

typedef struct {
  const char* name;
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v9_t* props);
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v9_t** sendDevComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm, ncclNetDeviceHandle_v9_t** recvDevComm);
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request);
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request);
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
  ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);
  ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
  ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_v9_t* props);
} ncclNet_v9_t;

/* ---- backend ids used by the mux ---- */
#define RR_BACKEND_SOCKET 0
#define RR_BACKEND_IB     1

/* ---- shared logging hook (set by plugin init, used by all backends) ---- */
extern ncclDebugLogger_t rrLog;
#define RR_INFO(fmt, ...) do { if (rrLog) rrLog(NCCL_LOG_INFO, 0, __FILE__, __LINE__, fmt, ##__VA_ARGS__); } while (0)
#define RR_WARN(fmt, ...) do { if (rrLog) rrLog(NCCL_LOG_WARN, 0, __FILE__, __LINE__, fmt, ##__VA_ARGS__); } while (0)

/* ---- handle layout shared between mux and backends ----
 * The 128-byte handle that NCCL exchanges is laid out by the mux as:
 *   [0]      uint8  backendsAvailable bitmask (bit0=socket, bit1=ib)
 *   [1..]    rrSockAddr  socket-backend listen address
 *   [..]     rrSockAddr  ib OOB listen address
 * Both sub-handles are a fixed rrSockAddr (see below), well under 128.
 */
typedef struct {
  uint32_t ip;    /* IPv4, network byte order */
  uint16_t port;  /* network byte order */
  uint16_t _pad;
} rrSockAddr;     /* 8 bytes */

/* The 128-byte handle the mux fills in listen() and the connector reads.
 * Exposed here (not private to plugin.c) so tests can construct one. */
typedef struct {
  uint8_t     backends;     /* bit0 socket, bit1 ib */
  uint8_t     _pad[3];
  rrSockAddr socketAddr;   /* socket-backend listen addr */
  rrSockAddr ibOobAddr;    /* ib OOB listen addr */
} rrHandle;

/* ---- socket backend public API (socket_net.c) ---- */
ncclResult_t rrSockListen(int dev, rrSockAddr* addr, void** listenComm);
ncclResult_t rrSockConnect(int dev, rrSockAddr* addr, void** sendComm);
ncclResult_t rrSockAccept(void* listenComm, void** recvComm);
ncclResult_t rrSockRegMr(void* comm, void* data, size_t size, int type, void** mhandle);
ncclResult_t rrSockDeregMr(void* comm, void* mhandle);
ncclResult_t rrSockIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request);
ncclResult_t rrSockIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request);
ncclResult_t rrSockIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t rrSockTest(void* request, int* done, int* sizes);
ncclResult_t rrSockCloseSend(void* sendComm);
ncclResult_t rrSockCloseRecv(void* recvComm);
ncclResult_t rrSockCloseListen(void* listenComm);

/* ---- IB backend public API (ib_net.c) ---- */
ncclResult_t rrIbInit(void);
ncclResult_t rrIbListen(int dev, rrSockAddr* addr, void** listenComm);
ncclResult_t rrIbConnect(int dev, rrSockAddr* addr, void** sendComm);
ncclResult_t rrIbAccept(void* listenComm, void** recvComm);
ncclResult_t rrIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle);
ncclResult_t rrIbDeregMr(void* comm, void* mhandle);
ncclResult_t rrIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request);
ncclResult_t rrIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request);
ncclResult_t rrIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t rrIbTest(void* request, int* done, int* sizes);
ncclResult_t rrIbCloseSend(void* sendComm);
ncclResult_t rrIbCloseRecv(void* recvComm);
ncclResult_t rrIbCloseListen(void* listenComm);

#endif /* RDMAROUTE_NET_ABI_H_ */
