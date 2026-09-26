import React from 'react';
import Mermaid from '@theme/Mermaid';
import diagrams from '@site/src/data/architecture.json';

/**
 * Renders every diagram in `src/data/architecture.json` via Docusaurus's
 * Mermaid theme component. Data-driven so a cloned sibling site only needs
 * to replace the JSON, never this component.
 */
export default function DiagramSection(): React.JSX.Element {
  return (
    <>
      {diagrams.map((d) => (
        <section key={d.id} id={d.id} style={{marginBottom: '3rem'}}>
          <h2>{d.title}</h2>
          <p>{d.description}</p>
          <Mermaid value={d.mermaid} />
        </section>
      ))}
    </>
  );
}
