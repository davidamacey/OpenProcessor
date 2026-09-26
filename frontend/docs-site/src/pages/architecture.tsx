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
          Three diagrams, kept accurate to <code>CLAUDE.md</code>: the system context (who
          talks to whom), the labeling workflow (the loop a dataset goes through), and the
          frontend's own internal structure. Rendered with Mermaid rather than a generated
          Archify export — diagram sources live in{' '}
          <code>docs-site/src/data/architecture.json</code>; edit that file when the code
          they describe changes.
        </p>
        <DiagramSection />
      </main>
    </Layout>
  );
}
