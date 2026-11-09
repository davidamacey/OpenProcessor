/**
 * The request of a served `next_steps` entry (`{action, method, path, reason}`).
 * The step names a route and a method but serves no body, while `POST
 * /clusters/runs` needs `{"target": ...}`: the `cluster_regions` step says in
 * its `reason` that the body is `{"target": "region_box"}`. Every other step is
 * sent without a body. A step that serves its own `body` would replace this.
 */
export function nextStepInit(step: { action: string; method: string }): RequestInit {
  const init: RequestInit = { method: step.method.toUpperCase() };
  if (step.action === 'cluster_regions') {
    init.body = JSON.stringify({ target: 'region_box' });
  }
  return init;
}
