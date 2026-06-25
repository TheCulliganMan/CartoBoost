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
    detail: 'Route between regression, forecasting, graph, neural, and utility APIs.',
    href: '/docs/user-guide/model-types',
  },
  {
    title: 'Model in the browser',
    detail: 'Drag in a taxi dataset and run forecasts, regression, graph, and neural modeling locally.',
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
  'Forecasting models',
  'Graph and neural surfaces',
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
          Temporal, spatial, geotemporal, and graph-aware regression for
          taxi-domain modeling, demand forecasting, and benchmark-backed
          experiments.
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
      <div className={styles.mapPanel} aria-label="CartoBoost route and signal illustration">
        <img src={require('@site/static/img/route-signal.svg').default} alt="" />
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
          <code>{`from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    splitters=["axis", "periodic:24", "gaussian_2d"],
)

model.fit(X_train, y_train)
predictions = model.predict(X_validation)`}</code>
        </pre>
        <Link to="/docs/benchmarks">Read the benchmark reports</Link>
      </div>
      <div className={styles.benchmarkPanel}>
        <img
          src={require('@site/docs/assets/model_benchmarks/mae_by_model.png').default}
          alt="Model benchmark MAE comparison chart"
        />
        <p>
          Benchmark pages connect plots to split design, model settings, and
          what the result means for taxi-style modeling.
        </p>
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
