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

export type MarketExplorerEdge = {
  source: string;
  target: string;
  weight: number;
  kinds: string[];
};

export type MarketStructureExplorerProps = {
  nodes: MarketExplorerNode[];
  edges: MarketExplorerEdge[];
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
};

/**
 * Converts the model's portable lane evidence into endpoint markets for the
 * interactive view.  This is deliberately a display projection: the native
 * lane-level payload remains the auditable source of truth.
 */
export function marketExplorerDataFromPayload(payload: MarketStructureExplorerPayload): MarketStructureExplorerProps {
  const laneId = (row: {lane_id?: string; laneId?: string}) => String(row.lane_id ?? row.laneId ?? '');
  const finite = (value: unknown) => Number.isFinite(Number(value)) ? Number(value) : 0;
  const nodes = payload.lanes.map((lane) => {
    const id = laneId(lane);
    const originId = String(lane.origin_id ?? lane.originId ?? '');
    const destinationId = String(lane.destination_id ?? lane.destinationId ?? '');
    const forecast = payload.forecasts.filter((row) => laneId(row) === id).sort((a, b) => a.horizon - b.horizon);
    const explanation = payload.explanations.find((row) => laneId(row) === id);
    const primary = finite(explanation?.smoothed_primary ?? explanation?.smoothedPrimary ?? forecast[0]?.primary);
    const path = forecast.map((row) => finite(row.primary));
    const inbound = payload.lanes
      .filter((candidate) => String(candidate.destination_id ?? candidate.destinationId ?? '') === originId)
      .flatMap((candidate) => payload.forecasts.filter((row) => laneId(row) === laneId(candidate)))
      .reduce((sum, row) => sum + finite(row.secondary), 0);
    const outbound = forecast.reduce((sum, row) => sum + finite(row.secondary), 0);
    return {
      id,
      label: `${originId} → ${destinationId}`,
      longitude: finite(lane.origin_x ?? lane.originX),
      latitude: finite(lane.origin_y ?? lane.originY),
      inbound,
      outbound,
      primary,
      primaryChange: primary ? (finite(path.at(-1)) - primary) / primary * 100 : 0,
      secondary: outbound,
      forecast: path.length ? path : [primary],
    };
  });
  const edges = payload.kernels.flatMap((edge) => {
    const source = edge.source_lane_id ?? edge.sourceLaneId;
    const target = edge.target_lane_id ?? edge.targetLaneId;
    return source && target ? [{source, target, weight: finite(edge.weight), kinds: edge.kinds}] : [];
  });
  return {nodes, edges};
}

type Metric = 'outbound' | 'inbound' | 'forecast_change' | 'kernel';

const metricLabel: Record<Metric, string> = {
  outbound: 'Outbound demand',
  inbound: 'Inbound demand',
  forecast_change: 'Forecast change',
  kernel: 'Learned kernel',
};

export default function MarketStructureExplorer({nodes, edges}: MarketStructureExplorerProps): React.ReactElement {
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<{setProps: (props: unknown) => void; finalize: () => void} | null>(null);
  const mapRef = useRef<{remove: () => void; fitBounds: (bounds: unknown, options: unknown) => void} | null>(null);
  const [metric, setMetric] = useState<Metric>('outbound');
  const [selectedId, setSelectedId] = useState(nodes[0]?.id ?? '');
  const [mapReady, setMapReady] = useState(false);
  const selected = nodes.find((node) => node.id === selectedId) ?? nodes[0];

  const selectedEdges = useMemo(
    () => edges.filter((edge) => edge.source === selected?.id || edge.target === selected?.id).sort((a, b) => b.weight - a.weight),
    [edges, selected?.id],
  );

  useEffect(() => {
    if (!mapContainer.current || nodes.length === 0) return undefined;
    let cancelled = false;
    void (async () => {
      const [{default: maplibregl}, {MapboxOverlay}, {ScatterplotLayer, ArcLayer, TextLayer}] = await Promise.all([
        import('maplibre-gl'),
        import('@deck.gl/mapbox'),
        import('@deck.gl/layers'),
      ]);
      if (cancelled || !mapContainer.current) return;
      const center = nodes.reduce<[number, number]>((sum, node) => [sum[0] + node.longitude / nodes.length, sum[1] + node.latitude / nodes.length], [0, 0]);
      const map = new maplibregl.Map({
        attributionControl: false,
        center,
        container: mapContainer.current,
        cooperativeGestures: true,
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        zoom: 10,
      });
      const byId = new Map(nodes.map((node) => [node.id, node]));
      const overlay = new MapboxOverlay({interleaved: false, layers: buildLayers({ArcLayer, ScatterplotLayer, TextLayer, byId, edges, metric, selectedId, onSelect: setSelectedId})});
      map.addControl(overlay);
      map.once('load', () => {
        const points = nodes.map((node) => [node.longitude, node.latitude] as [number, number]);
        const bounds = points.reduce((next, point) => next.extend(point), new maplibregl.LngLatBounds(points[0], points[0]));
        map.fitBounds(bounds, {duration: 0, padding: 54});
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
  }, [edges, nodes]);

  useEffect(() => {
    if (!overlayRef.current) return;
    void (async () => {
      const [{ScatterplotLayer, ArcLayer, TextLayer}] = await Promise.all([import('@deck.gl/layers')]);
      const byId = new Map(nodes.map((node) => [node.id, node]));
      overlayRef.current?.setProps({layers: buildLayers({ArcLayer, ScatterplotLayer, TextLayer, byId, edges, metric, selectedId, onSelect: setSelectedId})});
    })();
  }, [edges, metric, nodes, selectedId]);

  if (!selected) return <div className={styles.empty}>No market points available.</div>;
  const localEdges = selectedEdges.slice(0, 6);
  return (
    <section className={styles.explorer} aria-label="Interactive learned market structure explorer">
      <div className={styles.controls}>
        <label>
          View
          <select value={metric} onChange={(event) => setMetric(event.target.value as Metric)}>
            {(Object.keys(metricLabel) as Metric[]).map((value) => <option value={value} key={value}>{metricLabel[value]}</option>)}
          </select>
        </label>
        <span className={styles.hint}>{mapReady ? 'Select any point or kernel arc' : 'Loading map'}</span>
      </div>
      <div className={styles.map} ref={mapContainer} />
      <div className={styles.detail}>
        <div>
          <span className={styles.eyebrow}>Selected market</span>
          <strong>{selected.label}</strong>
          <span>{format(selected.outbound)} outbound · {format(selected.inbound)} inbound</span>
        </div>
        <div className={styles.metricGrid}>
          <Metric label="Forecast" value={format(selected.primary)} />
          <Metric label="Change" value={`${selected.primaryChange >= 0 ? '+' : ''}${selected.primaryChange.toFixed(1)}%`} />
          <Metric label="Supporting volume" value={format(selected.secondary)} />
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
  return <MarketStructureExplorer nodes={sampleNodes} edges={sampleEdges} />;
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

function buildLayers({ArcLayer, ScatterplotLayer, TextLayer, byId, edges, metric, selectedId, onSelect}: any): any[] {
  const nodes = [...byId.values()] as MarketExplorerNode[];
  const selectedEdges = edges.filter((edge: MarketExplorerEdge) => edge.source === selectedId || edge.target === selectedId);
  const weights = selectedEdges.map((edge: MarketExplorerEdge) => edge.weight);
  const maxWeight = Math.max(...weights, 1);
  const value = (node: MarketExplorerNode): number => metric === 'outbound' ? node.outbound : metric === 'inbound' ? node.inbound : metric === 'forecast_change' ? Math.abs(node.primaryChange) : selectedEdges.filter((edge: MarketExplorerEdge) => edge.source === node.id || edge.target === node.id).reduce((sum: number, edge: MarketExplorerEdge) => sum + edge.weight, 0);
  const maxValue = Math.max(...nodes.map(value), 1);
  return [
    new ArcLayer({id: 'market-kernels', data: selectedEdges, getSourcePosition: (edge: MarketExplorerEdge) => [byId.get(edge.source).longitude, byId.get(edge.source).latitude], getTargetPosition: (edge: MarketExplorerEdge) => [byId.get(edge.target).longitude, byId.get(edge.target).latitude], getSourceColor: [58, 182, 220, 220], getTargetColor: [244, 154, 77, 200], getWidth: (edge: MarketExplorerEdge) => 1 + edge.weight / maxWeight * 7, getHeight: 0.22, pickable: true, onClick: ({object}: any) => onSelect(byEdgePeer(object, selectedId))}),
    new ScatterplotLayer({id: 'market-points', data: nodes, getPosition: (node: MarketExplorerNode) => [node.longitude, node.latitude], getRadius: (node: MarketExplorerNode) => 7 + value(node) / maxValue * 16, radiusUnits: 'pixels', stroked: true, getLineColor: (node: MarketExplorerNode) => node.id === selectedId ? [255, 255, 255, 255] : [190, 208, 220, 180], getLineWidth: (node: MarketExplorerNode) => node.id === selectedId ? 3 : 1, getFillColor: (node: MarketExplorerNode) => node.id === selectedId ? [255, 138, 77, 240] : [59, 170, 211, 205], pickable: true, onClick: ({object}: any) => onSelect(object.id)}),
    new TextLayer({id: 'market-labels', data: nodes, getPosition: (node: MarketExplorerNode) => [node.longitude, node.latitude], getText: (node: MarketExplorerNode) => node.label, getColor: [235, 243, 248, 245], getSize: 12, getTextAnchor: 'middle', getAlignmentBaseline: 'top', getPixelOffset: [0, 17], fontFamily: 'system-ui'}),
  ];
}
