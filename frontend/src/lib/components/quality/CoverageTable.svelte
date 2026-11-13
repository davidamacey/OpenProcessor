<script lang="ts">
  /** Per-class coverage as served: how far a person has checked each class's items. */
  import DataTable, { type Column } from '$components/ui/DataTable.svelte';
  import type { ClassCoverage } from '$lib/types_quality';

  let { coverage }: { coverage: ClassCoverage[] } = $props();

  const columns: Column<ClassCoverage>[] = [
    { id: 'class_name', label: 'Class', value: (r) => r.class_name },
    { id: 'validated', label: 'Checked', value: (r) => r.validated, align: 'right' },
    {
      id: 'machine_only',
      label: 'Machine only',
      value: (r) => r.machine_only,
      align: 'right',
    },
    {
      id: 'detector_only',
      label: 'Detector only',
      value: (r) => r.detector_only,
      align: 'right',
    },
    { id: 'test', label: 'In test', value: (r) => r.test, align: 'right' },
  ];
  const cellText = (r: ClassCoverage, id: string): string =>
    id === 'class_name'
      ? r.class_name
      : (
          r[id as 'validated' | 'machine_only' | 'detector_only' | 'test'] ?? 0
        ).toLocaleString();
</script>

<div data-testid="readiness-coverage">
  <DataTable
    {columns}
    rows={coverage}
    rowKey={(r) => r.class_name}
    label="Coverage per class"
    maxHeightClass="max-h-96"
    initialSort={{ column: 'validated', dir: 'asc' }}
  >
    {#snippet cell(row, col)}{cellText(row, col.id)}{/snippet}
  </DataTable>
</div>
