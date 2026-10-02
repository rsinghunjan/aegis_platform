import { afterEach, describe, expect, it, vi } from "vitest";
import { aegisApi } from "./client";

describe("Aegis API client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses tenant-scoped control-plane routes and request contracts", async () => {
    const fetchMock = vi.fn().mockImplementation(
      () =>
        new Response(JSON.stringify({ items: [], offset: 0, limit: 50 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await aegisApi.listRuns("tenant/a");
    await aegisApi.listApprovals("tenant/a");
    await aegisApi.decideApproval("tenant/a", "approval/1", "approve");
    await aegisApi.listEvidence("tenant/a", "run/1");

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/operator/agent/runs?tenant_id=tenant%2Fa",
      "/agent/approvals?tenant_id=tenant%2Fa&status=pending",
      "/agent/approvals/approval%2F1/decision",
      "/agent/runs/run%2F1/evidence?tenant_id=tenant%2Fa",
    ]);
    expect(JSON.parse(fetchMock.mock.calls[2][1].body)).toEqual({
      tenant_id: "tenant/a",
      decision: "approve",
    });
  });
});
