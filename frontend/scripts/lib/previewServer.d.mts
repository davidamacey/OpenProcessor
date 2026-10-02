import type { ChildProcess, SpawnOptions } from 'node:child_process';

export function spawnGroup(
  cmd: string,
  args: string[],
  opts?: SpawnOptions,
): ChildProcess;
export function stopGroup(child: ChildProcess, graceMs?: number): Promise<void>;
export function stopGroupSync(child: ChildProcess, pauseMs?: number): void;
export function startPreview(root: string, port: number): ChildProcess;
