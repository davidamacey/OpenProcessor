/**
 * The request of a served `next_steps` entry (`{action, method, path, reason,
 * body}`). The backend serves the JSON `body` a step needs (null when none);
 * the client sends it as served and builds nothing itself.
 */
export function nextStepInit(step: { method: string; body?: unknown }): RequestInit {
  const init: RequestInit = { method: step.method.toUpperCase() };
  if (step.body != null) init.body = JSON.stringify(step.body);
  return init;
}
