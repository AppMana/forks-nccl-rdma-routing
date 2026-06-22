// SPDX-License-Identifier: Apache-2.0
/*
 * Proves the built plugin satisfies the NCCL v9 net-plugin ABI contract:
 * dlopen the .so, resolve the `ncclNetPlugin_v9` symbol NCCL looks for, and
 * assert the name + every function pointer is present. This is what NCCL itself
 * does when it loads libnccl-net.so.
 *
 * build+run: cc -O2 -Wall -Isrc -o /tmp/vt tests/vtable_test.c -ldl && /tmp/vt ./libnccl-net-rdmaroute.so
 */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include "net_abi.h"

static int failures;
#define CHECK(cond, msg) do { \
	if (cond) printf("  [ok] %s\n", (msg)); \
	else { printf("  [FAIL] %s\n", (msg)); failures++; } \
} while (0)

int main(int argc, char **argv)
{
	const char *so = argc > 1 ? argv[1] : "./libnccl-net-rdmaroute.so";
	void *h = dlopen(so, RTLD_NOW | RTLD_LOCAL);
	CHECK(h != NULL, "dlopen the plugin .so");
	if (!h) { printf("  dlerror: %s\nFAILED (1 failures)\n", dlerror()); return 1; }

	ncclNet_v9_t *p = (ncclNet_v9_t *)dlsym(h, "ncclNetPlugin_v9");
	CHECK(p != NULL, "ncclNetPlugin_v9 symbol present (what NCCL dlsym's)");
	if (!p) { printf("FAILED (1 failures)\n"); return 1; }

	CHECK(p->name && strcmp(p->name, "rdmaroute") == 0, "plugin name is 'rdmaroute'");

	/* every v9 entry point must be non-NULL */
	CHECK(p->init && p->devices && p->getProperties, "init/devices/getProperties present");
	CHECK(p->listen && p->connect && p->accept, "listen/connect/accept present");
	CHECK(p->regMr && p->regMrDmaBuf && p->deregMr, "regMr/regMrDmaBuf/deregMr present");
	CHECK(p->isend && p->irecv && p->iflush && p->test, "isend/irecv/iflush/test present");
	CHECK(p->closeSend && p->closeRecv && p->closeListen, "close* present");
	CHECK(p->getDeviceMr && p->irecvConsumed && p->makeVDevice, "device-offload entry points present");

	/* devices() must report exactly one virtual NIC */
	p->init(NULL);
	int ndev = -1;
	CHECK(p->devices(&ndev) == ncclSuccess && ndev == 1, "devices() reports one rdmaroute NIC");

	dlclose(h);
	printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
	return failures ? 1 : 0;
}
