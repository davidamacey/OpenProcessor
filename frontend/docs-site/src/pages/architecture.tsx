import React from 'react';
import Layout from '@theme/Layout';
import DiagramSection from '@site/src/components/DiagramSection';

export default function Architecture(): React.JSX.Element {
  return (
    <Layout
      title="Architecture"
      description="Cropwright architecture diagrams: system context, labeling workflow, and frontend internals.">
      <main className="container margin-vert--lg">
        <h1>Architecture</h1>
        <p>
          Three views of the system: who talks to whom, the loop a dataset goes through, and
          how the frontend is put together. The Mermaid sources live in{' '}
          <code>docs-site/src/data/architecture.json</code>.
        </p>
        <DiagramSection />
      </main>
    </Layout>
  );
}
