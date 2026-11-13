/**
 * The served name of the object page being shown (a file name, an import name), for the last
 * breadcrumb. The page that loads the object publishes it with the path it belongs to, so a
 * stale name never shows on another page.
 */
export const crumbName = $state<{ path: string; label: string }>({ path: '', label: '' });

export function setCrumbName(path: string, label: string): void {
  crumbName.path = path;
  crumbName.label = label;
}
