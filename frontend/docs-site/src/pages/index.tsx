import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

import Hero from '@site/src/components/Hero';
import FeatureGrid from '@site/src/components/FeatureGrid';
import HowItWorks from '@site/src/components/HowItWorks';
import QuickStart from '@site/src/components/QuickStart';
import ScreenshotShowcase from '@site/src/components/ScreenshotShowcase';

export default function Home(): React.JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <Hero />
      <main>
        <FeatureGrid />
        <HowItWorks />
        <ScreenshotShowcase />
        <QuickStart />
      </main>
    </Layout>
  );
}
