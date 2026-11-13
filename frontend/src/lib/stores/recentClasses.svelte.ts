/**
 * The classes the user assigned last in this project, most recent first: the class picker ranks
 * them first and binds the keys 1 to 9 to them (`class_picker.recent_N`), so a class used a
 * moment ago is one key. Held in memory for the session; a project switch drops it.
 */
import { onProjectChange } from '$lib/projectChange';

export const RECENT_CLASS_LIMIT = 9;

class RecentClassesStore {
  names = $state<string[]>([]);

  note(name: string): void {
    this.names = [name, ...this.names.filter((n) => n !== name)].slice(
      0,
      RECENT_CLASS_LIMIT,
    );
  }

  reset(): void {
    this.names = [];
  }
}

export const recentClassesStore = new RecentClassesStore();
onProjectChange(() => recentClassesStore.reset());
