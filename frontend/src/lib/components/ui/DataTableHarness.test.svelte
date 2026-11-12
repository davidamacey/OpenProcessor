<script lang="ts">
  import DataTable, { type Column } from './DataTable.svelte';

  interface Row {
    name: string;
    n: number;
  }
  const rows: Row[] = [
    { name: 'cog', n: 9 },
    { name: 'bolt', n: 2 },
    { name: 'person', n: 30 },
  ];
  const columns: Column<Row>[] = [
    { id: 'name', label: 'Class', value: (r) => r.name },
    { id: 'n', label: 'Items', value: (r) => r.n, align: 'right' },
    { id: 'note', label: 'Note' },
  ];
</script>

<DataTable {columns} {rows} rowKey={(r) => r.name} label="Test table">
  {#snippet cell(row, col)}
    {col.id === 'name' ? row.name : col.id === 'n' ? row.n : ''}
  {/snippet}
</DataTable>
