/**
 * Assist on /clusters/[id]: opens the dialog for the viewed cluster or
 * sub-cluster (and the crops selected in it), and loads the newest run that
 * produced a label proposal for that scope. The server lists the runs and
 * says, per run, whether the scope has changed since (`scope_now`).
 */
import { listAssistRuns } from '$lib/api_assist';
import { apiErrorText } from '$lib/api_errors';
import { AssistRunController } from '$lib/assistRun/assistRunController.svelte';
import type { AssistRunList, AssistRunState } from '$lib/types_assist';

interface ClusterView {
  clusterId: number;
  subTab: string | null;
  sel: { ids: Iterable<string> };
}

interface Deps {
  list: (scope: {
    clusterId: number;
    clusterSubid: string | null;
  }) => Promise<AssistRunList>;
}

const DEFAULT_DEPS: Deps = { list: (scope) => listAssistRuns(scope) };

export function createClusterAssist(view: ClusterView, deps: Deps = DEFAULT_DEPS) {
  const assist = new AssistRunController();
  let latest = $state<AssistRunState | null>(null);
  let loadError = $state<string | null>(null);
  let seq = 0;

  function openAssist(): void {
    const subid = view.subTab;
    const ids = [...view.sel.ids];
    assist.open({
      title: subid ? `Sub-cluster ${subid}` : `Cluster #${view.clusterId}`,
      selected:
        ids.length > 0
          ? {
              kind: 'crops',
              crop_ids: ids,
              context: { cluster_id: view.clusterId, cluster_subid: subid },
            }
          : undefined,
      whole: { kind: 'cluster', cluster_id: view.clusterId, cluster_subid: subid },
    });
  }

  async function load(): Promise<void> {
    const mine = ++seq;
    try {
      const res = await deps.list({
        clusterId: view.clusterId,
        clusterSubid: view.subTab,
      });
      if (mine !== seq) return;
      latest = res.runs.find((r) => r.result?.proposal) ?? null;
      loadError = null;
    } catch (e) {
      if (mine !== seq) return;
      latest = null;
      loadError = apiErrorText(e);
    }
  }

  return {
    assist,
    get latest() {
      return latest;
    },
    get loadError() {
      return loadError;
    },
    openAssist,
    load,
  };
}

export type ClusterAssist = ReturnType<typeof createClusterAssist>;
