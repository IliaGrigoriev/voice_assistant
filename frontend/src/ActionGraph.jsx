import { useEffect, useState } from "react";

const NODE_WIDTH = 156;
const NODE_HEIGHT = 72;
const BASE_COLUMN_GAP = 120;
const ROW_GAP = 152;
const PADDING = 76;

function shorten(label, maxLength = 18) {
  if (label.length <= maxLength) return label;
  return `${label.slice(0, maxLength - 1)}…`;
}

function boxesOverlap(left, right, gap = 10) {
  return !(
    left.x + left.width + gap <= right.x ||
    right.x + right.width + gap <= left.x ||
    left.y + left.height + gap <= right.y ||
    right.y + right.height + gap <= left.y
  );
}

function createEdgeLayouts(edges, positions, width, height) {
  const occupied = [...positions.values()].map((position) => ({
    x: position.x,
    y: position.y,
    width: NODE_WIDTH,
    height: NODE_HEIGHT,
  }));

  return edges.map((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return null;

    const isHorizontal = Math.abs(source.y - target.y) < NODE_HEIGHT;
    const horizontalDirection = target.x >= source.x ? 1 : -1;
    const verticalDirection = target.y >= source.y ? 1 : -1;
    const sourceX = isHorizontal
      ? source.x + (horizontalDirection > 0 ? NODE_WIDTH : 0)
      : source.x + NODE_WIDTH / 2;
    const sourceY = isHorizontal
      ? source.y + NODE_HEIGHT / 2
      : source.y + (verticalDirection > 0 ? NODE_HEIGHT : 0);
    const targetX = isHorizontal
      ? target.x + (horizontalDirection > 0 ? 0 : NODE_WIDTH)
      : target.x + NODE_WIDTH / 2;
    const targetY = isHorizontal
      ? target.y + NODE_HEIGHT / 2
      : target.y + (verticalDirection > 0 ? 0 : NODE_HEIGHT);
    const middleX = (sourceX + targetX) / 2;
    const middleY = (sourceY + targetY) / 2;
    const relation = String(edge.relation ?? "relationship").replaceAll("_", " ");
    const labelWidth = Math.max(54, relation.length * 7 + 20);
    const path = isHorizontal
      ? `M ${sourceX} ${sourceY} C ${middleX} ${sourceY}, ${middleX} ${targetY}, ${targetX} ${targetY}`
      : `M ${sourceX} ${sourceY} C ${sourceX} ${middleY}, ${targetX} ${middleY}, ${targetX} ${targetY}`;
    const initialX = middleX;
    const initialY = isHorizontal ? middleY - 24 : middleY;
    const offsets = [{ x: 0, y: 0 }];

    for (let distance = 34; distance <= 340; distance += 34) {
      offsets.push(
        { x: distance, y: 0 },
        { x: -distance, y: 0 },
        { x: 0, y: distance },
        { x: 0, y: -distance },
        { x: distance, y: distance },
        { x: -distance, y: distance },
        { x: distance, y: -distance },
        { x: -distance, y: -distance },
      );
    }

    const availablePosition = offsets.find((offset) => {
      const box = {
        x: initialX + offset.x - labelWidth / 2,
        y: initialY + offset.y - 13,
        width: labelWidth,
        height: 26,
      };
      const insideCanvas =
        box.x >= 12 &&
        box.y >= 12 &&
        box.x + box.width <= width - 12 &&
        box.y + box.height <= height - 12;
      return insideCanvas && occupied.every((item) => !boxesOverlap(box, item));
    }) ?? { x: 0, y: 0 };

    const labelX = initialX + availablePosition.x;
    const labelY = initialY + availablePosition.y;
    occupied.push({
      x: labelX - labelWidth / 2,
      y: labelY - 13,
      width: labelWidth,
      height: 26,
    });

    return { edge, labelWidth, labelX, labelY, path, relation };
  }).filter(Boolean);
}

function createLayout(graph) {
  const nodes = graph.nodes ?? [];
  const edges = graph.edges ?? [];
  const levels = new Map();
  const queue = [];
  const nodeIds = new Set(nodes.map((node) => node.id));
  const targetedIds = new Set(edges.map((edge) => edge.target));
  const inferredRoot = nodes.find((node) => !targetedIds.has(node.id))?.id;
  const rootId = nodeIds.has(graph.root) ? graph.root : inferredRoot ?? nodes[0]?.id;

  if (rootId) {
    levels.set(rootId, 0);
    queue.push(rootId);
  }

  while (queue.length) {
    const source = queue.shift();
    const nextLevel = levels.get(source) + 1;

    edges
      .filter((edge) => edge.source === source)
      .forEach((edge) => {
        if (!levels.has(edge.target)) {
          levels.set(edge.target, nextLevel);
          queue.push(edge.target);
        }
      });
  }

  const deepestLevel = Math.max(0, ...levels.values());
  nodes.forEach((node) => {
    if (!levels.has(node.id)) levels.set(node.id, deepestLevel + 1);
  });

  const rows = new Map();
  nodes.forEach((node) => {
    const level = levels.get(node.id);
    if (!rows.has(level)) rows.set(level, []);
    rows.get(level).push(node);
  });

  const widestRow = Math.max(1, ...[...rows.values()].map((row) => row.length));
  const longestLabelWidth = Math.max(
    0,
    ...edges.map((edge) => String(edge.relation ?? "relationship").length * 7 + 20),
  );
  const columnGap = Math.max(BASE_COLUMN_GAP, longestLabelWidth + 32);
  const width = Math.max(
    900,
    PADDING * 2 + widestRow * NODE_WIDTH + (widestRow - 1) * columnGap,
  );
  const rowCount = Math.max(1, rows.size);
  const contentHeight = rowCount * NODE_HEIGHT + (rowCount - 1) * ROW_GAP;
  const height = Math.max(
    520,
    PADDING * 2 + contentHeight,
  );
  const startY = (height - contentHeight) / 2;
  const positions = new Map();

  [...rows.entries()]
    .sort(([left], [right]) => left - right)
    .forEach(([level, row], rowIndex) => {
      const rowWidth = row.length * NODE_WIDTH + (row.length - 1) * columnGap;
      const startX = (width - rowWidth) / 2;

      row.forEach((node, columnIndex) => {
        positions.set(node.id, {
          x: startX + columnIndex * (NODE_WIDTH + columnGap),
          y: startY + rowIndex * (NODE_HEIGHT + ROW_GAP),
        });
      });
    });

  const edgeLayouts = createEdgeLayouts(edges, positions, width, height);
  return { edgeLayouts, edges, height, nodes, positions, rootId, width };
}

export default function ActionGraph({ graph }) {
  const [zoom, setZoom] = useState(1);
  const layout = createLayout(graph);

  useEffect(() => {
    setZoom(1);
  }, [graph]);

  function changeZoom(amount) {
    setZoom((current) => Math.min(2, Math.max(0.5, Number((current + amount).toFixed(2)))));
  }

  if (!layout.nodes.length) {
    return <p className="empty-state">No graph nodes were returned.</p>;
  }

  return (
    <div className="graph-visual">
      <div className="graph-toolbar">
        <div className="graph-legend" aria-hidden="true">
          <span><i className="legend-dot action" />Action</span>
          <span><i className="legend-dot entity" />Entity</span>
          <span><i className="legend-dot quantity" />Quantity</span>
          <span><i className="legend-dot unit" />Unit</span>
          <span><i className="legend-dot time" />Time</span>
        </div>

        <div className="zoom-controls" aria-label="Graph zoom controls">
          <button
            type="button"
            aria-label="Zoom out"
            disabled={zoom <= 0.5}
            onClick={() => changeZoom(-0.15)}
          >
            −
          </button>
          <button
            className="zoom-level"
            type="button"
            aria-label="Reset zoom"
            onClick={() => setZoom(1)}
          >
            {Math.round(zoom * 100)}%
          </button>
          <button
            type="button"
            aria-label="Zoom in"
            disabled={zoom >= 2}
            onClick={() => changeZoom(0.15)}
          >
            +
          </button>
        </div>
      </div>

      <div className="graph-canvas">
        <div
          className="graph-stage"
          style={{
            width: `${layout.width * zoom}px`,
            height: `${layout.height * zoom}px`,
          }}
        >
          <svg
            width={layout.width * zoom}
            height={layout.height * zoom}
            viewBox={`0 0 ${layout.width} ${layout.height}`}
            role="img"
            aria-label={`Action graph with ${layout.nodes.length} nodes and ${layout.edges.length} connections`}
          >
          <defs>
            <marker
              id="arrowhead"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" />
            </marker>
          </defs>

          <g className="graph-edges">
            {layout.edgeLayouts.map((item, index) => {
              const { edge, labelWidth, labelX, labelY, path, relation } = item;
              return (
                <g key={`${edge.source}-${edge.target}-${index}`}>
                  <path d={path} markerEnd="url(#arrowhead)" />
                  <g className="graph-edge-label" transform={`translate(${labelX} ${labelY})`}>
                    <rect x={-labelWidth / 2} y="-13" width={labelWidth} height="26" rx="8" />
                    <text y="4">{relation}</text>
                  </g>
                </g>
              );
            })}
          </g>

          <g className="graph-nodes">
            {layout.nodes.map((node) => {
              const position = layout.positions.get(node.id);
              const nodeType = [
                "action",
                "entity",
                "measurement",
                "quantity",
                "time",
                "unit",
                "value",
              ].includes(node.type)
                ? node.type
                : "entity";
              const isRoot = node.id === layout.rootId;

              return (
                <g
                  key={node.id}
                  className={`graph-node graph-node--${nodeType}${isRoot ? " is-root" : ""}`}
                  transform={`translate(${position.x} ${position.y})`}
                >
                  <title>{`${node.label} (${node.type})`}</title>
                  <rect width={NODE_WIDTH} height={NODE_HEIGHT} rx="16" />
                  <text className="node-label" x={NODE_WIDTH / 2} y="29">
                    {shorten(String(node.label))}
                  </text>
                  <text className="node-type" x={NODE_WIDTH / 2} y="48">
                    {isRoot ? `${nodeType} · root` : nodeType}
                  </text>
                </g>
              );
            })}
          </g>
          </svg>
        </div>
      </div>
    </div>
  );
}
