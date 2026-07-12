import clsx from 'clsx';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

import styles from './index.module.css';

const paths = [
  {
    title: 'Train a tabular model',
    detail: 'Fit a structured regressor for taxi duration, fare, demand, or residual targets.',
    href: '/docs/getting-started',
  },
  {
    title: 'Choose a modeling path',
    detail: 'Start with structured regression, panel forecasting, or advanced relationship models.',
    href: '/docs/user-guide/model-types',
  },
  {
    title: 'Model in the browser',
    detail: 'Load a bundled sample and inspect structured regression or forecasting locally.',
    href: '/modeling-lab',
  },
  {
    title: 'Validate the evidence',
    detail: 'Read benchmark results by target, split, baseline, metric, and recommendation.',
    href: '/docs/benchmarks',
  },
];

const capabilities = [
  'Periodic time splitters',
  'Spatial and route-aware trees',
  'Sparse zone memberships',
  'Leakage-aware validation',
  'Panel forecasting',
  'Portable model artifacts',
];

function HomepageHeader() {
  return (
    <header className={styles.hero}>
      <div className={styles.heroText}>
        <span className={styles.eyebrow}>Temporal-spatial Python modeling</span>
        <Heading as="h1" className={styles.heroTitle}>
          CartoBoost
        </Heading>
        <p className={styles.heroSubtitle}>
          Rust-backed Python modeling for structured tabular and panel data where
          geometry, cyclic time, memberships, or direction matter.
        </p>
        <div className={styles.heroActions}>
          <Link className="button button--primary button--lg" to="/docs/installation">
            Install CartoBoost
          </Link>
          <Link
            className={clsx('button button--secondary button--lg', styles.heroSecondaryButton)}
            to="/docs/user-guide/model-types"
          >
            Choose a modeling path
          </Link>
        </div>
      </div>
    </header>
  );
}

function PathCards() {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <span className={styles.eyebrow}>Start with your task</span>
        <Heading as="h2">Use case first, reference second</Heading>
      </div>
      <div className={styles.cardGrid}>
        {paths.map((path) => (
          <Link className={styles.pathCard} to={path.href} key={path.title}>
            <Heading as="h3">{path.title}</Heading>
            <p>{path.detail}</p>
          </Link>
        ))}
      </div>
    </section>
  );
}

function CapabilityStrip() {
  return (
    <section className={clsx(styles.section, styles.signalBand)}>
      <div>
        <span className={styles.eyebrow}>Modeling primitives</span>
        <Heading as="h2">Built for place, time, and direction</Heading>
      </div>
      <ul className={styles.capabilityList}>
        {capabilities.map((capability) => (
          <li key={capability}>{capability}</li>
        ))}
      </ul>
    </section>
  );
}

function CodeAndEvidence() {
  return (
    <section className={clsx(styles.section, styles.splitSection)}>
      <div>
        <span className={styles.eyebrow}>First fit</span>
        <Heading as="h2">A small model before the deep dive</Heading>
        <pre className={styles.codeSample}>
          <code>{`# Run the complete dependency-free example:
python examples/quickstart.py

# Or start from the estimator directly:
from cartoboost import CartoBoostRegressor
model = CartoBoostRegressor(splitters=["axis", "periodic:24"])
model.fit(X_train, y_train)
predictions = model.predict(X_validation)`}</code>
        </pre>
        <Link to="/docs/getting-started">Run the complete quickstart</Link>
      </div>
      <div className={styles.benchmarkPanel}>
        <p>
          Benchmark pages connect current-code results to split design, model
          settings, runtime, and what the result means for taxi-style modeling.
        </p>
        <Link to="/docs/benchmarks">Read the benchmark reports</Link>
      </div>
    </section>
  );
}

export default function Home(): React.ReactElement {
  return (
    <Layout
      title="CartoBoost"
      description="Temporal, spatial, geotemporal, and graph-aware regression documentation"
    >
      <HomepageHeader />
      <main>
        <PathCards />
        <CapabilityStrip />
        <CodeAndEvidence />
      </main>
    </Layout>
  );
}
