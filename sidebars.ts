import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    {
      type: 'category',
      label: 'Get Started',
      collapsed: false,
      items: [
        'index',
        'installation',
        'getting-started',
        'user-guide/model-types',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Tabular Modeling',
          items: [
            'user-guide/python-estimator',
            'user-guide/classification-quickstart',
            'user-guide/ranking-quickstart',
            'user-guide/categorical-features',
            'user-guide/spatial-cv-best-practices',
            'user-guide/parameters',
            'objectives',
            'constraints',
            'spatial_modeling',
            'feature_schema',
            'sparse_features',
            'shap',
            'model_artifact',
          ],
        },
        {
          type: 'category',
          label: 'Forecasting',
          collapsed: false,
          items: [
            'forecasting',
            'user-guide/forecasting-models/index',
            'user-guide/forecasting-models/auto-forecaster',
            'user-guide/forecasting-models/naive-seasonal',
            'user-guide/forecasting-models/theta',
            'user-guide/forecasting-models/ets',
            'user-guide/forecasting-models/arima',
            'user-guide/forecasting-models/kalman',
            'user-guide/forecasting-models/piecewise-linear-seasonal',
            'user-guide/forecasting-models/kriging',
            'user-guide/forecasting-models/spatial-piecewise-kriging',
            'user-guide/forecasting-models/cartoboost-lag',
            'user-guide/forecasting-models/ensembles',
          ],
        },
        'graph-features',
        'neural-features',
        'general_utilities',
        'plotting',
        'user-guide/cli',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: ['reference/python-api', 'reference/cli', 'feature_catalog'],
    },
    {
      type: 'category',
      label: 'Benchmarks',
      collapsed: false,
      items: [
        'benchmarks/index',
        'benchmarks/model-suite',
        'benchmarks/methodology',
        'benchmarks/nyc-taxi',
        'benchmarks/forecasting',
        'benchmarks/taxi-zone',
        'benchmarks/lane-level',
        'benchmarks/neural-embedding-benchmark-latest',
      ],
    },
  ],
};

export default sidebars;
