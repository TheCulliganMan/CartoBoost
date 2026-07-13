import {useEffect, useMemo, useRef, useState} from 'react';
import 'maplibre-gl/dist/maplibre-gl.css';

import styles from './styles.module.css';

export type MarketExplorerNode = {
  id: string;
  label: string;
  longitude: number;
  latitude: number;
  inbound: number;
  outbound: number;
  primary: number;
  primaryChange: number;
  secondary: number;
  forecast: number[];
};

type H3MarketNode = MarketExplorerNode & {
  h3Cell: string;
  boundary: [number, number][];
};

export type MarketExplorerEdge = {
  source: string;
  target: string;
  weight: number;
  kinds: string[];
};

export type MarketStructureExplorerProps = {
  nodes: MarketExplorerNode[];
  edges: MarketExplorerEdge[];
  /** Human-readable names for the model's two target channels, when supplied. */
  targetNames?: string[];
};

/** The JSON-compatible contract returned by the Rust model in Python and WASM. */
export type MarketStructureExplorerPayload = {
  lanes: Array<{
    lane_id?: string; laneId?: string;
    origin_id?: string; originId?: string;
    destination_id?: string; destinationId?: string;
    origin_x?: number; originX?: number;
    origin_y?: number; originY?: number;
    destination_x?: number; destinationX?: number;
    destination_y?: number; destinationY?: number;
  }>;
  forecasts: Array<{lane_id?: string; laneId?: string; horizon: number; primary: number; secondary: number}>;
  explanations: Array<{
    lane_id?: string; laneId?: string;
    smoothed_primary?: number; smoothedPrimary?: number;
    observed_primary?: number | null; observedPrimary?: number | null;
    top_relationships?: Array<{source_lane_id?: string; sourceLaneId?: string; target_lane_id?: string; targetLaneId?: string; weight: number; kinds: string[]}>;
  }>;
  kernels: Array<{source_lane_id?: string; sourceLaneId?: string; target_lane_id?: string; targetLaneId?: string; weight: number; kinds: string[]}>;
  target_names?: string[]; targetNames?: string[];
};

/**
 * Converts the model's portable lane evidence into endpoint markets for the
 * interactive view.  This is deliberately a display projection: the native
 * lane-level payload remains the auditable source of truth.
 */
export function marketExplorerDataFromPayload(payload: MarketStructureExplorerPayload): MarketStructureExplorerProps {
  const laneId = (row: {lane_id?: string; laneId?: string}) => String(row.lane_id ?? row.laneId ?? '');
  const finite = (value: unknown) => Number.isFinite(Number(value)) ? Number(value) : 0;
  const requireRows = <T,>(value: unknown, name: string): T[] => {
    if (!Array.isArray(value)) throw new Error(`Invalid market explorer payload: ${name} must be an array.`);
    return value as T[];
  };
  const lanes = requireRows<MarketStructureExplorerPayload['lanes'][number]>(payload.lanes, 'lanes');
  const forecasts = requireRows<MarketStructureExplorerPayload['forecasts'][number]>(payload.forecasts, 'forecasts');
  const explanations = requireRows<MarketStructureExplorerPayload['explanations'][number]>(payload.explanations, 'explanations');
  const kernels = requireRows<MarketStructureExplorerPayload['kernels'][number]>(payload.kernels, 'kernels');
  const lanesById = new Map(lanes.map((lane) => [laneId(lane), lane]));
  const forecastsByLane = new Map<string, Array<{horizon: number; primary: number; secondary: number}>>();
  forecasts.forEach((row) => {
    const id = laneId(row);
    const values = forecastsByLane.get(id) ?? [];
    values.push({horizon: row.horizon, primary: finite(row.primary), secondary: finite(row.secondary)});
    forecastsByLane.set(id, values);
  });
  forecastsByLane.forEach((rows) => rows.sort((left, right) => left.horizon - right.horizon));
  const explanationsByLane = new Map(explanations.map((row) => [laneId(row), row]));
  const marketLanes = new Map<string, typeof lanes>();
  lanes.forEach((lane) => {
    const origin = String(lane.origin_id ?? lane.originId ?? '');
    const rows = marketLanes.get(origin) ?? [];
    rows.push(lane);
    marketLanes.set(origin, rows);
  });
  const nodes = [...marketLanes.entries()].map(([originId, lanes]) => {
    const horizonCount = Math.max(1, ...lanes.map((lane) => forecastsByLane.get(laneId(lane))?.length ?? 0));
    const forecast = Array.from({length: horizonCount}, (_, index) => weightedMean(lanes.map((lane) => {
      const row = forecastsByLane.get(laneId(lane))?.[index];
      return {value: row?.primary ?? 0, weight: Math.max(row?.secondary ?? 0, 1)};
    })));
    const current = weightedMean(lanes.map((lane) => {
      const rows = forecastsByLane.get(laneId(lane)) ?? [];
      const explanation = explanationsByLane.get(laneId(lane));
      return {
        value: finite(explanation?.smoothed_primary ?? explanation?.smoothedPrimary ?? rows[0]?.primary),
        weight: Math.max(rows[0]?.secondary ?? 0, 1),
      };
    }));
    const outbound = lanes.reduce((sum, lane) => sum + (forecastsByLane.get(laneId(lane)) ?? []).reduce((subtotal, row) => subtotal + row.secondary, 0), 0);
    const inbound = lanes
      .filter((lane) => String(lane.destination_id ?? lane.destinationId ?? '') === originId)
      .reduce((sum, lane) => sum + (forecastsByLane.get(laneId(lane)) ?? []).reduce((subtotal, row) => subtotal + row.secondary, 0), 0);
    return {
      id: originId,
      label: `Market ${originId}`,
      longitude: weightedMean(lanes.map((lane) => ({value: finite(lane.origin_x ?? lane.originX), weight: 1}))),
      latitude: weightedMean(lanes.map((lane) => ({value: finite(lane.origin_y ?? lane.originY), weight: 1}))),
      inbound,
      outbound,
      primary: current,
      primaryChange: current ? (forecast.at(-1)! - current) / current * 100 : 0,
      secondary: outbound,
      forecast: forecast.length ? forecast : [current],
    };
  }).sort((left, right) => left.id.localeCompare(right.id));
  const marketEdges = new Map<string, MarketExplorerEdge>();
  kernels.forEach((edge) => {
    const sourceLane = lanesById.get(String(edge.source_lane_id ?? edge.sourceLaneId ?? ''));
    const targetLane = lanesById.get(String(edge.target_lane_id ?? edge.targetLaneId ?? ''));
    const source = sourceLane && String(sourceLane.origin_id ?? sourceLane.originId ?? '');
    const target = targetLane && String(targetLane.origin_id ?? targetLane.originId ?? '');
    if (!source || !target || source === target) return;
    const key = `${source}\u0000${target}`;
    const current = marketEdges.get(key);
    marketEdges.set(key, {
      source,
      target,
      weight: Math.max(current?.weight ?? 0, finite(edge.weight)),
      kinds: [...new Set([...(current?.kinds ?? []), ...(edge.kinds ?? [])])],
    });
  });
  const edges = [...marketEdges.values()].sort((left, right) => right.weight - left.weight);
  return {nodes, edges, targetNames: payload.target_names ?? payload.targetNames};
}

type Metric = 'demand' | 'price' | 'forecast_change' | 'kernel';
type SurfaceMode = 'kernel' | 'grid';

const metricLabel: Record<Metric, string> = {
  demand: 'Demand surface',
  price: 'Primary target surface',
  forecast_change: 'Forecast change',
  kernel: 'Network support',
};

export default function MarketStructureExplorer({nodes, edges, targetNames}: MarketStructureExplorerProps): React.ReactElement {
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<{setProps: (props: unknown) => void; finalize: () => void} | null>(null);
  const mapRef = useRef<{remove: () => void; fitBounds: (bounds: unknown, options: unknown) => void; setMaxBounds: (bounds: unknown) => void; setMinZoom: (zoom: number) => void; setMaxZoom: (zoom: number) => void; getZoom: () => number} | null>(null);
  const [h3Nodes, setH3Nodes] = useState<H3MarketNode[]>([]);
  const [metric, setMetric] = useState<Metric>('demand');
  const [surfaceMode, setSurfaceMode] = useState<SurfaceMode>('kernel');
  const [selectedId, setSelectedId] = useState(nodes[0]?.id ?? '');
  const [mapReady, setMapReady] = useState(false);
  const selected = nodes.find((node) => node.id === selectedId) ?? nodes[0];
  const labels = metricLabels(targetNames);

  useEffect(() => {
    let cancelled = false;
    void import('h3-js').then((h3) => {
      if (!cancelled) setH3Nodes(nodes.map((node) => h3MarketNode(node, h3)));
    }).catch(() => {
      if (!cancelled) setH3Nodes([]);
    });
    return () => { cancelled = true; };
  }, [nodes]);

  const selectedEdges = useMemo(
    () => edges.filter((edge) => edge.source === selected?.id || edge.target === selected?.id).sort((a, b) => b.weight - a.weight),
    [edges, selected?.id],
  );

  useEffect(() => {
    if (!mapContainer.current || h3Nodes.length === 0) return undefined;
    let cancelled = false;
    void (async () => {
      const [{default: maplibregl}, {MapboxOverlay}, {ArcLayer, TextLayer}, {H3HexagonLayer}] = await Promise.all([
        import('maplibre-gl'),
        import('@deck.gl/mapbox'),
        import('@deck.gl/layers'),
        import('@deck.gl/geo-layers'),
      ]);
      if (cancelled || !mapContainer.current) return;
      const center = h3Nodes.reduce<[number, number]>((sum, node) => [sum[0] + node.longitude / h3Nodes.length, sum[1] + node.latitude / h3Nodes.length], [0, 0]);
      const map = new maplibregl.Map({
        attributionControl: false,
        center,
        container: mapContainer.current,
        cooperativeGestures: true,
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        zoom: 10,
      });
      const byId = new Map(h3Nodes.map((node) => [node.id, node]));
      const overlay = new MapboxOverlay({interleaved: false, layers: buildLayers({ArcLayer, H3HexagonLayer, TextLayer, byId, edges, metric, surfaceMode, selectedId, onSelect: setSelectedId})});
      map.addControl(overlay);
      map.once('load', () => {
        const points = h3Nodes.flatMap((node) => node.boundary);
        const bounds = points.reduce((next, point) => next.extend(point), new maplibregl.LngLatBounds(points[0], points[0]));
        map.fitBounds(bounds, {duration: 0, padding: 54});
        map.setMaxBounds(bounds);
        map.setMinZoom(Math.max(7, map.getZoom() - 1));
        map.setMaxZoom(Math.min(14, map.getZoom() + 3));
        if (!cancelled) setMapReady(true);
      });
      overlayRef.current = overlay;
      mapRef.current = map;
    })().catch(() => setMapReady(false));
    return () => {
      cancelled = true;
      overlayRef.current?.finalize();
      mapRef.current?.remove();
      overlayRef.current = null;
      mapRef.current = null;
    };
  }, [edges, h3Nodes]);

  useEffect(() => {
    if (!overlayRef.current) return;
    void (async () => {
      const [{ArcLayer, TextLayer}, {H3HexagonLayer}] = await Promise.all([import('@deck.gl/layers'), import('@deck.gl/geo-layers')]);
      const byId = new Map(h3Nodes.map((node) => [node.id, node]));
      overlayRef.current?.setProps({layers: buildLayers({ArcLayer, H3HexagonLayer, TextLayer, byId, edges, metric, surfaceMode, selectedId, onSelect: setSelectedId})});
    })();
  }, [edges, h3Nodes, metric, selectedId, surfaceMode]);

  if (!selected) return <div className={styles.empty}>No market points available.</div>;
  const localEdges = selectedEdges.slice(0, 6);
  return (
    <section className={styles.explorer} aria-label="Interactive learned market structure explorer">
      <div className={styles.controls}>
        <label>
          Signal
          <select value={metric} onChange={(event) => setMetric(event.target.value as Metric)}>
            {(Object.keys(metricLabel) as Metric[]).map((value) => <option value={value} key={value}>{labels[value]}</option>)}
          </select>
        </label>
        <label>
          Geometry
          <select value={surfaceMode} onChange={(event) => setSurfaceMode(event.target.value as SurfaceMode)}>
            <option value="kernel">H3 prediction cells</option>
            <option value="grid">H3 prediction extrusions</option>
          </select>
        </label>
        <span className={styles.hint}>{mapReady ? `${surfaceMode === 'kernel' ? 'H3 prediction field' : 'H3 prediction volume'} · select a market or relationship` : 'Loading map'}</span>
      </div>
      <div className={styles.mapShell}>
        <div className={styles.map} ref={mapContainer} />
        <div className={styles.legend} aria-label={`${labels[metric]} color scale`}>
          <span>{surfaceMode === 'kernel' ? 'Low' : 'Low volume'}</span><i /><span>{surfaceMode === 'kernel' ? 'High' : 'High volume'}</span>
        </div>
        <div className={styles.surfaceNote}><strong>{labels[metric]}</strong><span>{surfaceMode === 'kernel' ? 'Model prediction, assigned to H3 resolution 9 cells' : 'Model prediction, H3 cell height = signal'}</span></div>
      </div>
      <div className={styles.detail}>
        <div>
          <span className={styles.eyebrow}>Selected market</span>
          <strong>{selected.label}</strong>
          <span>{format(selected.outbound)} outbound · {format(selected.inbound)} inbound</span>
        </div>
        <div className={styles.metricGrid}>
          <Metric label={targetNames?.[0] ?? 'Price signal'} value={format(selected.primary)} />
          <Metric label="Change" value={`${selected.primaryChange >= 0 ? '+' : ''}${selected.primaryChange.toFixed(1)}%`} />
          <Metric label={targetNames?.[1] ?? 'Demand volume'} value={format(selected.secondary)} />
        </div>
      </div>
      <div className={styles.bottomGrid}>
        <div>
          <span className={styles.eyebrow}>Forecast path</span>
          <ForecastLine values={selected.forecast} />
        </div>
        <div>
          <span className={styles.eyebrow}>Top learned relationships</span>
          <ol className={styles.relationships}>
            {localEdges.map((edge) => {
              const peer = byEdgePeer(edge, selected.id);
              const peerNode = nodes.find((node) => node.id === peer);
              return <li key={`${edge.source}:${edge.target}`}><button type="button" onClick={() => setSelectedId(peer)}>{peerNode?.label ?? peer}</button><span>{edge.weight.toFixed(2)} · {edge.kinds.join(', ')}</span></li>;
            })}
          </ol>
        </div>
      </div>
    </section>
  );
}

/** Interactive taxi-domain sample; production callers pass their artifact-backed rows. */
export function MarketStructureExplorerSample(): React.ReactElement {
  return <MarketStructureExplorer nodes={sampleNodes} edges={sampleEdges} targetNames={['Fare / price index', 'Trip demand']} />;
}

type H3TaxiCell = {id: string; longitude: number; latitude: number; boundary: [number, number][]; baseDemand: number};
type H3TaxiLane = {id: string; source: H3TaxiCell; target: H3TaxiCell; baseline: number};

/**
 * A deliberately high-cardinality visual exercise for directional taxi flow.
 * It is a deterministic NYC-wide scenario, not a substitution for TLC data or
 * a benchmark result. Applications can use the same H3 lane shape with their
 * artifact-backed origin/destination flows.
 */
export function H3TaxiFlowDemo(): React.ReactElement {
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<{setProps: (props: unknown) => void; finalize: () => void} | null>(null);
  const mapRef = useRef<{remove: () => void} | null>(null);
  // Build after mount so this expensive, map-only network stays out of the
  // server-rendered document and first contentful paint.
  const [network, setNetwork] = useState<{cells: H3TaxiCell[]; lanes: H3TaxiLane[]}>({cells: [], lanes: []});
  const [hour, setHour] = useState(17);
  const [minimumTrips, setMinimumTrips] = useState(28);
  const [selectedId, setSelectedId] = useState('');
  const [mapReady, setMapReady] = useState(false);
  const selected = network.cells.find((cell) => cell.id === selectedId);
  const activeLanes = useMemo(() => network.lanes
    .map((lane) => ({...lane, trips: flowAtHour(lane, hour)}))
    .filter((lane) => lane.trips >= minimumTrips)
    .sort((left, right) => right.trips - left.trips), [hour, minimumTrips, network.lanes]);
  const renderedLanes = useMemo(() => {
    const focused = selectedId
      ? activeLanes.filter((lane) => lane.source.id === selectedId || lane.target.id === selectedId)
      : activeLanes;
    return focused.slice(0, selectedId ? 700 : 520);
  }, [activeLanes, selectedId]);
  const selectedOutbound = selectedId ? activeLanes.filter((lane) => lane.source.id === selectedId).reduce((sum, lane) => sum + lane.trips, 0) : 0;
  const selectedInbound = selectedId ? activeLanes.filter((lane) => lane.target.id === selectedId).reduce((sum, lane) => sum + lane.trips, 0) : 0;

  useEffect(() => {
    let cancelled = false;
    let frame = 0;
    void import('h3-js').then((h3) => {
      frame = window.requestAnimationFrame(() => {
        if (!cancelled) setNetwork(buildH3TaxiDemoNetwork(h3));
      });
    });
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
    };
  }, []);

  useEffect(() => {
    if (!mapContainer.current || network.cells.length === 0) return undefined;
    let cancelled = false;
    void (async () => {
      const [{default: maplibregl}, {MapboxOverlay}, {ArcLayer, PolygonLayer, ScatterplotLayer}] = await Promise.all([
        import('maplibre-gl'), import('@deck.gl/mapbox'), import('@deck.gl/layers'),
      ]);
      if (cancelled || !mapContainer.current) return;
      const map = new maplibregl.Map({
        attributionControl: false,
        center: [-73.92, 40.71],
        container: mapContainer.current,
        cooperativeGestures: true,
        scrollZoom: false,
        dragRotate: false,
        touchPitch: false,
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        zoom: 9.7,
      });
      const overlay = new MapboxOverlay({interleaved: false, layers: buildH3TaxiLayers({ArcLayer, PolygonLayer, ScatterplotLayer, cells: network.cells, lanes: renderedLanes, selectedId, onSelect: setSelectedId})});
      map.addControl(overlay);
      map.once('load', () => { if (!cancelled) setMapReady(true); });
      overlayRef.current = overlay;
      mapRef.current = map;
    })().catch(() => setMapReady(false));
    return () => {
      cancelled = true;
      overlayRef.current?.finalize();
      mapRef.current?.remove();
      overlayRef.current = null;
      mapRef.current = null;
    };
  }, [network.cells]);

  useEffect(() => {
    if (!overlayRef.current) return;
    void (async () => {
      const [{ArcLayer, PolygonLayer, ScatterplotLayer}] = await Promise.all([import('@deck.gl/layers')]);
      overlayRef.current?.setProps({layers: buildH3TaxiLayers({ArcLayer, PolygonLayer, ScatterplotLayer, cells: network.cells, lanes: renderedLanes, selectedId, onSelect: setSelectedId})});
    })();
  }, [network.cells, renderedLanes, selectedId]);

  return <section className={styles.h3Explorer} aria-label="NYC H3 directional taxi flow demonstration">
    <div className={styles.h3Controls}>
      <label>Hour <strong>{formatHour(hour)}</strong><input aria-label="Hour of day" type="range" min="0" max="23" value={hour} onChange={(event) => setHour(Number(event.target.value))} /></label>
      <label>Minimum trips <strong>{minimumTrips}</strong><input aria-label="Minimum trips per directional lane" type="range" min="10" max="100" step="2" value={minimumTrips} onChange={(event) => setMinimumTrips(Number(event.target.value))} /></label>
      <span className={styles.h3Hint}>{mapReady ? `${network.cells.length.toLocaleString()} H3 cells · ${network.lanes.length.toLocaleString()} directional lanes · ${renderedLanes.length.toLocaleString()} drawn` : 'Loading H3 flow map'}</span>
    </div>
    <div className={styles.h3MapShell}>
      <div className={styles.map} ref={mapContainer} />
      <div className={styles.h3Legend}><span>Pickup hex intensity</span><i /><span>Destination</span></div>
      <div className={styles.h3Note}><strong>{selected ? `H3 ${selected.id.slice(0, 8)}` : 'NYC-wide directional lane field'}</strong><span>{selected ? `${format(selectedOutbound)} outbound · ${format(selectedInbound)} inbound at ${formatHour(hour)}` : 'Click a pickup or dropoff hex to isolate its lanes'}</span></div>
    </div>
    <p className={styles.h3Caption}>Illustrative H3 network for interaction and rendering scale; it is not a TLC trip-count claim. The lane shape matches an artifact-backed pickup-cell → dropoff-cell flow table.</p>
  </section>;
}

const sampleNodes: MarketExplorerNode[] = [
  {id: '132', label: 'JFK', longitude: -73.7781, latitude: 40.6413, inbound: 610, outbound: 980, primary: 4.82, primaryChange: 7.4, secondary: 980, forecast: [4.82, 4.9, 5.03, 5.16, 5.11, 5.2, 5.28]},
  {id: '138', label: 'LaGuardia', longitude: -73.874, latitude: 40.7769, inbound: 770, outbound: 895, primary: 4.61, primaryChange: 4.2, secondary: 895, forecast: [4.61, 4.65, 4.7, 4.77, 4.81, 4.86, 4.9]},
  {id: '161', label: 'Midtown', longitude: -73.9857, latitude: 40.758, inbound: 1560, outbound: 1330, primary: 3.94, primaryChange: 2.1, secondary: 1330, forecast: [3.94, 3.98, 4.01, 4.05, 4.02, 4.08, 4.1]},
  {id: '230', label: 'Times Sq.', longitude: -73.987, latitude: 40.7589, inbound: 1190, outbound: 1260, primary: 4.12, primaryChange: -1.5, secondary: 1260, forecast: [4.12, 4.1, 4.06, 4.08, 4.05, 4.02, 4.01]},
  {id: '48', label: 'Financial Dist.', longitude: -74.009, latitude: 40.7075, inbound: 870, outbound: 640, primary: 3.72, primaryChange: 3.8, secondary: 640, forecast: [3.72, 3.78, 3.81, 3.85, 3.83, 3.88, 3.91]},
  {id: '79', label: 'East Village', longitude: -73.985, latitude: 40.727, inbound: 540, outbound: 590, primary: 3.58, primaryChange: 0.7, secondary: 590, forecast: [3.58, 3.6, 3.59, 3.62, 3.64, 3.65, 3.67]},
];

const sampleEdges: MarketExplorerEdge[] = [
  {source: '132', target: '138', weight: 0.91, kinds: ['geographic', 'neural_kernel']},
  {source: '132', target: '161', weight: 0.78, kinds: ['shared_destination', 'residual_correlation']},
  {source: '132', target: '230', weight: 0.65, kinds: ['residual_correlation']},
  {source: '138', target: '161', weight: 0.84, kinds: ['geographic', 'neural_kernel']},
  {source: '161', target: '230', weight: 0.88, kinds: ['shared_origin', 'neural_kernel']},
  {source: '161', target: '48', weight: 0.69, kinds: ['residual_correlation']},
  {source: '230', target: '79', weight: 0.73, kinds: ['geographic']},
  {source: '48', target: '79', weight: 0.61, kinds: ['reverse_lane']},
];

function Metric({label, value}: {label: string; value: string}) {
  return <span><small>{label}</small><strong>{value}</strong></span>;
}

function ForecastLine({values}: {values: number[]}) {
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  const points = values.map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 100},${36 - ((value - min) / range) * 30}`).join(' ');
  return <svg className={styles.forecast} viewBox="0 0 100 42" role="img" aria-label="Selected market forecast path"><polyline points={points} /><circle cx="100" cy={36 - ((values.at(-1)! - min) / range) * 30} r="2.5" /></svg>;
}

function byEdgePeer(edge: MarketExplorerEdge, selected: string): string { return edge.source === selected ? edge.target : edge.source; }
function format(value: number): string { return Intl.NumberFormat('en-US', {maximumFractionDigits: 0}).format(value); }
function weightedMean(values: Array<{value: number; weight: number}>): number {
  const totalWeight = values.reduce((sum, row) => sum + Math.max(0, row.weight), 0);
  if (totalWeight === 0) return values.length ? values.reduce((sum, row) => sum + row.value, 0) / values.length : 0;
  return values.reduce((sum, row) => sum + row.value * Math.max(0, row.weight), 0) / totalWeight;
}

function h3MarketNode(node: MarketExplorerNode, h3: typeof import('h3-js')): H3MarketNode {
  const h3Cell = h3.latLngToCell(node.latitude, node.longitude, 9);
  const rawBoundary = h3.cellToBoundary(h3Cell, true) as [number, number][];
  const first = rawBoundary[0];
  const last = rawBoundary.at(-1);
  const boundary = last?.[0] === first?.[0] && last?.[1] === first?.[1] ? rawBoundary : [...rawBoundary, first];
  return {...node, h3Cell, boundary};
}
function metricLabels(targetNames?: string[]): Record<Metric, string> {
  const primary = targetNames?.[0]?.trim();
  const secondary = targetNames?.[1]?.trim();
  return {
    demand: secondary ? `${secondary} surface` : metricLabel.demand,
    price: primary ? `${primary} surface` : metricLabel.price,
    forecast_change: primary ? `${primary} change` : metricLabel.forecast_change,
    kernel: metricLabel.kernel,
  };
}
function marketSignal(node: MarketExplorerNode, edges: MarketExplorerEdge[], metric: Metric): number {
  if (metric === 'demand') return node.outbound;
  if (metric === 'price') return node.primary;
  if (metric === 'forecast_change') return Math.abs(node.primaryChange);
  return edges.filter((edge) => edge.source === node.id || edge.target === node.id).reduce((sum, edge) => sum + edge.weight, 0);
}

const NYC_H3_DEMO_CELLS = ['872a10000ffffff', '872a10001ffffff', '872a10002ffffff', '872a10003ffffff', '872a10004ffffff', '872a10006ffffff', '872a10008ffffff', '872a10009ffffff', '872a1000affffff', '872a1000bffffff', '872a1000dffffff', '872a1000effffff', '872a10010ffffff', '872a10011ffffff', '872a10012ffffff', '872a10013ffffff', '872a10014ffffff', '872a10015ffffff', '872a10016ffffff', '872a10018ffffff', '872a10019ffffff', '872a1001affffff', '872a1001bffffff', '872a1001cffffff', '872a1001dffffff', '872a1001effffff', '872a10030ffffff', '872a10031ffffff', '872a10032ffffff', '872a10033ffffff', '872a10035ffffff', '872a10042ffffff', '872a10043ffffff', '872a10046ffffff', '872a1004affffff', '872a10050ffffff', '872a10051ffffff', '872a10052ffffff', '872a10053ffffff', '872a10054ffffff', '872a10055ffffff', '872a10056ffffff', '872a10058ffffff', '872a10059ffffff', '872a1005affffff', '872a1005bffffff', '872a1005cffffff', '872a1005dffffff', '872a1005effffff', '872a10070ffffff', '872a10072ffffff', '872a10073ffffff', '872a10076ffffff', '872a10080ffffff', '872a10081ffffff', '872a10082ffffff', '872a10083ffffff', '872a10084ffffff', '872a10085ffffff', '872a10086ffffff', '872a10088ffffff', '872a10089ffffff', '872a1008affffff', '872a1008bffffff', '872a1008cffffff', '872a1008dffffff', '872a1008effffff', '872a10095ffffff', '872a10099ffffff', '872a1009cffffff', '872a1009dffffff', '872a100a0ffffff', '872a100a1ffffff', '872a100a2ffffff', '872a100a3ffffff', '872a100a4ffffff', '872a100a5ffffff', '872a100a6ffffff', '872a100a8ffffff', '872a100a9ffffff', '872a100aaffffff', '872a100abffffff', '872a100acffffff', '872a100adffffff', '872a100aeffffff', '872a100b1ffffff', '872a100b3ffffff', '872a100c0ffffff', '872a100c1ffffff', '872a100c2ffffff', '872a100c3ffffff', '872a100c4ffffff', '872a100c5ffffff', '872a100c6ffffff', '872a100c8ffffff', '872a100c9ffffff', '872a100caffffff', '872a100cbffffff', '872a100ccffffff', '872a100cdffffff', '872a100ceffffff', '872a100d0ffffff', '872a100d1ffffff', '872a100d2ffffff', '872a100d3ffffff', '872a100d4ffffff', '872a100d5ffffff', '872a100d6ffffff', '872a100d8ffffff', '872a100d9ffffff', '872a100daffffff', '872a100dbffffff', '872a100dcffffff', '872a100ddffffff', '872a100deffffff', '872a100e0ffffff', '872a100e1ffffff', '872a100e2ffffff', '872a100e3ffffff', '872a100e4ffffff', '872a100e5ffffff', '872a100e6ffffff', '872a100e8ffffff', '872a100e9ffffff', '872a100eaffffff', '872a100ebffffff', '872a100ecffffff', '872a100edffffff', '872a100eeffffff', '872a100f0ffffff', '872a100f1ffffff', '872a100f2ffffff', '872a100f3ffffff', '872a100f4ffffff', '872a100f5ffffff', '872a100f6ffffff', '872a10189ffffff', '872a1018bffffff', '872a10236ffffff', '872a102b4ffffff', '872a102b6ffffff', '872a10380ffffff', '872a10381ffffff', '872a10382ffffff', '872a10383ffffff', '872a10384ffffff', '872a10385ffffff', '872a10386ffffff', '872a10388ffffff', '872a1038affffff', '872a1038cffffff', '872a1038dffffff', '872a1038effffff', '872a10390ffffff', '872a10391ffffff', '872a10392ffffff', '872a10393ffffff', '872a10394ffffff', '872a10395ffffff', '872a10396ffffff', '872a10398ffffff', '872a10399ffffff', '872a1039affffff', '872a1039bffffff', '872a1039cffffff', '872a1039dffffff', '872a1039effffff', '872a103a0ffffff', '872a103a1ffffff', '872a103a2ffffff', '872a103a3ffffff', '872a103a4ffffff', '872a103a5ffffff', '872a103a6ffffff', '872a103a8ffffff', '872a103a9ffffff', '872a103aaffffff', '872a103abffffff', '872a103acffffff', '872a103aeffffff', '872a103b0ffffff', '872a103b1ffffff', '872a103b2ffffff', '872a103b3ffffff', '872a103b4ffffff', '872a103b5ffffff', '872a103b6ffffff', '872a10600ffffff', '872a10601ffffff', '872a10602ffffff', '872a10603ffffff', '872a10604ffffff', '872a10605ffffff', '872a10606ffffff', '872a10608ffffff', '872a10609ffffff', '872a1060affffff', '872a1060bffffff', '872a1060cffffff', '872a1060dffffff', '872a1060effffff', '872a10610ffffff', '872a10611ffffff', '872a10613ffffff', '872a10614ffffff', '872a10615ffffff', '872a10616ffffff', '872a10618ffffff', '872a10619ffffff', '872a1061affffff', '872a1061bffffff', '872a1061cffffff', '872a1061dffffff', '872a1061effffff', '872a10620ffffff', '872a10621ffffff', '872a10622ffffff', '872a10623ffffff', '872a10624ffffff', '872a10625ffffff', '872a10626ffffff', '872a10628ffffff', '872a10629ffffff', '872a1062affffff', '872a1062bffffff', '872a1062cffffff', '872a1062dffffff', '872a1062effffff', '872a10630ffffff', '872a10631ffffff', '872a10632ffffff', '872a10633ffffff', '872a10634ffffff', '872a10635ffffff', '872a10636ffffff', '872a10654ffffff', '872a10656ffffff', '872a10662ffffff', '872a10670ffffff', '872a10671ffffff', '872a10672ffffff', '872a10673ffffff', '872a10674ffffff', '872a10675ffffff', '872a10676ffffff', '872a106e0ffffff', '872a106e2ffffff', '872a106e4ffffff', '872a106e5ffffff', '872a106e6ffffff', '872a106f5ffffff', '872a10705ffffff', '872a10708ffffff', '872a10709ffffff', '872a1070affffff', '872a1070bffffff', '872a1070cffffff', '872a1070dffffff', '872a10719ffffff', '872a1071bffffff', '872a10720ffffff', '872a10721ffffff', '872a10723ffffff', '872a10724ffffff', '872a10725ffffff', '872a10728ffffff', '872a10729ffffff', '872a1072affffff', '872a1072bffffff', '872a1072cffffff', '872a1072dffffff', '872a1072effffff', '872a10740ffffff', '872a10741ffffff', '872a10742ffffff', '872a10743ffffff', '872a10744ffffff', '872a10745ffffff', '872a10746ffffff', '872a10748ffffff', '872a1074cffffff', '872a1074dffffff', '872a1074effffff', '872a10750ffffff', '872a10751ffffff', '872a10752ffffff', '872a10753ffffff', '872a10754ffffff', '872a10755ffffff', '872a10756ffffff', '872a10758ffffff', '872a1075affffff', '872a1075bffffff', '872a1075cffffff', '872a1075effffff', '872a10760ffffff', '872a10761ffffff', '872a10762ffffff', '872a10763ffffff', '872a10764ffffff', '872a10765ffffff', '872a10766ffffff', '872a10768ffffff', '872a10769ffffff', '872a1076affffff', '872a1076bffffff', '872a1076cffffff', '872a1076dffffff', '872a1076effffff', '872a10770ffffff', '872a10771ffffff', '872a10772ffffff', '872a10773ffffff', '872a10774ffffff', '872a10775ffffff', '872a10776ffffff', '872a10788ffffff', '872a10789ffffff', '872a1078dffffff'];

function buildH3TaxiDemoNetwork(h3: typeof import('h3-js')): {cells: H3TaxiCell[]; lanes: H3TaxiLane[]} {
  // Resolution-7 cell ids are fixed at build time from the five-borough
  // footprint. This avoids any browser binding initialization race while
  // retaining genuine H3 geometry for each pickup and dropoff market.
  const cells = NYC_H3_DEMO_CELLS.map((id) => {
    const [latitude, longitude] = h3.cellToLatLng(id);
    return {
      id,
      longitude,
      latitude,
      boundary: h3.cellToBoundary(id, true) as [number, number][],
      baseDemand: 18 + hashUnit(id) * 74 + centralityBoost(latitude, longitude),
    };
  });
  const lanes = cells.flatMap((source, sourceIndex) => Array.from({length: 14}, (_, rank) => {
    let targetIndex = (sourceIndex + 19 + rank * 47 + Math.floor(hashUnit(`${source.id}:${rank}`) * (cells.length - 1))) % cells.length;
    if (targetIndex === sourceIndex) targetIndex = (targetIndex + 1) % cells.length;
    const target = cells[targetIndex];
    const distance = Math.hypot(source.latitude - target.latitude, (source.longitude - target.longitude) * 0.76);
    return {
      id: `${source.id}:${target.id}`,
      source,
      target,
      baseline: Math.max(8, (source.baseDemand * (0.42 + hashUnit(`${target.id}:${source.id}`))) / (1 + distance * 13)),
    };
  }));
  return {cells, lanes};
}

function centralityBoost(latitude: number, longitude: number): number {
  const midtown = Math.exp(-((latitude - 40.758) ** 2 + (longitude + 73.985) ** 2) / 0.0013) * 95;
  const downtown = Math.exp(-((latitude - 40.708) ** 2 + (longitude + 74.008) ** 2) / 0.0018) * 52;
  const airports = Math.exp(-((latitude - 40.641) ** 2 + (longitude + 73.778) ** 2) / 0.002) * 50
    + Math.exp(-((latitude - 40.776) ** 2 + (longitude + 73.874) ** 2) / 0.0014) * 38;
  return midtown + downtown + airports;
}

function hashUnit(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 0xffffffff;
}

function flowAtHour(lane: H3TaxiLane, hour: number): number {
  const commute = Math.exp(-((hour - 8) ** 2) / 8) + 1.25 * Math.exp(-((hour - 17.5) ** 2) / 11);
  const night = 0.38 * Math.exp(-((hour - 1) ** 2) / 10);
  const direction = 0.78 + hashUnit(`${lane.id}:${hour}`) * 0.44;
  return Math.max(1, Math.round(lane.baseline * (0.58 + commute + night) * direction));
}

function formatHour(hour: number): string {
  const suffix = hour >= 12 ? 'PM' : 'AM';
  return `${hour % 12 || 12}:00 ${suffix}`;
}

function buildH3TaxiLayers({ArcLayer, PolygonLayer, ScatterplotLayer, cells, lanes, selectedId, onSelect}: any): any[] {
  const maxDemand = Math.max(...cells.map((cell: H3TaxiCell) => cell.baseDemand), 1);
  const maxTrips = Math.max(...lanes.map((lane: H3TaxiLane & {trips: number}) => lane.trips), 1);
  const selectedLanes = selectedId ? lanes.filter((lane: H3TaxiLane) => lane.source.id === selectedId || lane.target.id === selectedId) : lanes;
  return [
    new PolygonLayer({
      id: 'taxi-h3-cells', data: cells, getPolygon: (cell: H3TaxiCell) => cell.boundary,
      getFillColor: (cell: H3TaxiCell) => cell.id === selectedId ? [255, 205, 77, 175] : [22, 116, 159, 28 + Math.round(cell.baseDemand / maxDemand * 105)],
      getLineColor: (cell: H3TaxiCell) => cell.id === selectedId ? [255, 240, 190, 255] : [87, 193, 218, 105],
      getLineWidth: (cell: H3TaxiCell) => cell.id === selectedId ? 2.5 : 0.5, lineWidthUnits: 'pixels', stroked: true, filled: true,
      pickable: true, onClick: ({object}: {object?: H3TaxiCell}) => object && onSelect(object.id),
    }),
    new ArcLayer({
      id: 'taxi-h3-flows', data: selectedLanes,
      getSourcePosition: (lane: H3TaxiLane) => [lane.source.longitude, lane.source.latitude],
      getTargetPosition: (lane: H3TaxiLane) => [lane.target.longitude, lane.target.latitude],
      getSourceColor: [60, 197, 222, 215], getTargetColor: [255, 139, 73, 230],
      getWidth: (lane: H3TaxiLane & {trips: number}) => 0.8 + lane.trips / maxTrips * 4.6,
      getHeight: 0.22, pickable: true, onClick: ({object}: {object?: H3TaxiLane}) => object && onSelect(object.source.id),
    }),
    new ScatterplotLayer({
      id: 'taxi-h3-centers', data: cells, getPosition: (cell: H3TaxiCell) => [cell.longitude, cell.latitude],
      getRadius: (cell: H3TaxiCell) => 1.5 + cell.baseDemand / maxDemand * 4.5, radiusUnits: 'pixels',
      getFillColor: (cell: H3TaxiCell) => cell.id === selectedId ? [255, 236, 177, 255] : [118, 226, 219, 165],
      pickable: true, onClick: ({object}: {object?: H3TaxiCell}) => object && onSelect(object.id),
    }),
  ];
}

function buildLayers({ArcLayer, H3HexagonLayer, TextLayer, byId, edges, metric, surfaceMode, selectedId, onSelect}: any): any[] {
  const nodes = [...byId.values()] as H3MarketNode[];
  const selectedEdges = edges.filter((edge: MarketExplorerEdge) => edge.source === selectedId || edge.target === selectedId);
  const weights = selectedEdges.map((edge: MarketExplorerEdge) => edge.weight);
  const maxWeight = Math.max(...weights, 1);
  const value = (node: MarketExplorerNode): number => marketSignal(node, edges, metric);
  const maxValue = Math.max(...nodes.map(value), 1);
  const fill = (node: H3MarketNode): [number, number, number, number] => {
    if (node.id === selectedId) return [255, 138, 77, 245];
    const intensity = Math.min(1, Math.max(0, value(node) / maxValue));
    return intensity < 0.5
      ? [24, Math.round(112 + intensity * 94), Math.round(154 + intensity * 55), 190]
      : [Math.round(29 + (intensity - .5) * 442), Math.round(159 + (intensity - .5) * 88), Math.round(190 - (intensity - .5) * 244), 215];
  };
  return [
    new ArcLayer({id: 'market-kernels', data: selectedEdges, getSourcePosition: (edge: MarketExplorerEdge) => [byId.get(edge.source).longitude, byId.get(edge.source).latitude], getTargetPosition: (edge: MarketExplorerEdge) => [byId.get(edge.target).longitude, byId.get(edge.target).latitude], getSourceColor: [58, 182, 220, 220], getTargetColor: [244, 154, 77, 200], getWidth: (edge: MarketExplorerEdge) => 1 + edge.weight / maxWeight * 7, getHeight: 0.22, pickable: true, onClick: ({object}: any) => onSelect(byEdgePeer(object, selectedId))}),
    new H3HexagonLayer({id: 'market-h3-prediction-cells', data: nodes, getHexagon: (node: H3MarketNode) => node.h3Cell, extruded: surfaceMode === 'grid', getElevation: (node: H3MarketNode) => surfaceMode === 'grid' ? 120 + value(node) / maxValue * 1600 : 0, getFillColor: fill, getLineColor: (node: H3MarketNode) => node.id === selectedId ? [255, 247, 210, 255] : [151, 229, 225, 205], getLineWidth: (node: H3MarketNode) => node.id === selectedId ? 3 : 1, lineWidthUnits: 'pixels', stroked: true, filled: true, pickable: true, onClick: ({object}: {object?: H3MarketNode}) => object && onSelect(object.id)}),
    new TextLayer({id: 'market-h3-labels', data: nodes, getPosition: (node: H3MarketNode) => [node.longitude, node.latitude], getText: (node: H3MarketNode) => node?.h3Cell?.slice(0, 8) ?? '', getColor: [235, 243, 248, 245], getSize: 11, getTextAnchor: 'middle', getAlignmentBaseline: 'center', fontFamily: 'system-ui'}),
  ];
}
