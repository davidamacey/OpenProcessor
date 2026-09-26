import React from 'react';
import Layout from '@theme/Layout';
import RoadmapView from '@site/src/components/RoadmapView';

export default function Roadmap(): React.JSX.Element {
  return (
    <Layout title="Roadmap" description="What's shipped in Cropwright and what's planned next.">
      <main className="container margin-vert--lg">
        <h1>Roadmap</h1>
        <RoadmapView />
      </main>
    </Layout>
  );
}
