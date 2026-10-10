import React from "react";

function safeLinkTarget(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:", "mailto:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function renderInline(text, keyPrefix) {
  const pattern = /(\*\*.+?\*\*|__.+?__|`[^`]+`|\*[^*\n]+\*|_[^_\n]+_|\[[^\]]+\]\((?:https?:\/\/|mailto:)[^)]+\))/g;
  return text.split(pattern).filter(Boolean).map((part, index) => {
    const key = `${keyPrefix}-${index}`;
    if ((part.startsWith("**") && part.endsWith("**")) || (part.startsWith("__") && part.endsWith("__"))) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*") || part.startsWith("_") && part.endsWith("_")) {
      return <em key={key}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code className="inline-code" key={key}>{part.slice(1, -1)}</code>;
    }
    const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) {
      const target = safeLinkTarget(link[2]);
      if (target) return <a key={key} href={target} target="_blank" rel="noreferrer">{link[1]}</a>;
    }
    return part;
  });
}

function isListItem(line) {
  return /^\s*[-+*]\s+/.test(line) || /^\s*\d+[.)]\s+/.test(line);
}

function isBlockStart(lines, index) {
  const line = lines[index];
  return /^\s{0,3}#{1,6}\s+/.test(line)
    || /^\s*```/.test(line)
    || /^\s*>/.test(line)
    || /^\s*(---+|___+|\*\*\*+)\s*$/.test(line)
    || isListItem(line)
    || (line.includes("|") && /^\s*\|?\s*:?-{3,}/.test(lines[index + 1] || ""));
}

function parseTableRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

export default function FormattedMessage({ content }) {
  const lines = String(content || "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }

    const fence = line.match(/^\s*```\s*([\w-]*)/);
    if (fence) {
      const code = [];
      index += 1;
      while (index < lines.length && !/^\s*```/.test(lines[index])) code.push(lines[index++]);
      if (index < lines.length) index += 1;
      blocks.push(<pre key={`block-${blocks.length}`}><code className={fence[1] ? `language-${fence[1]}` : undefined}>{code.join("\n")}</code></pre>);
      continue;
    }

    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const Tag = `h${Math.min(heading[1].length, 4)}`;
      blocks.push(<Tag key={`block-${blocks.length}`}>{renderInline(heading[2], `heading-${blocks.length}`)}</Tag>);
      index += 1;
      continue;
    }

    if (/^\s*(---+|___+|\*\*\*+)\s*$/.test(line)) {
      blocks.push(<hr key={`block-${blocks.length}`} />);
      index += 1;
      continue;
    }

    if (isListItem(line)) {
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      const items = [];
      while (index < lines.length && isListItem(lines[index]) === true && /^\s*\d+[.)]\s+/.test(lines[index]) === ordered) {
        const text = lines[index].replace(ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-+*]\s+/, "");
        items.push(<li key={`item-${items.length}`}>{renderInline(text, `list-${blocks.length}-${items.length}`)}</li>);
        index += 1;
      }
      const List = ordered ? "ol" : "ul";
      blocks.push(<List key={`block-${blocks.length}`}>{items}</List>);
      continue;
    }

    if (line.includes("|") && /^\s*\|?\s*:?-{3,}/.test(lines[index + 1] || "")) {
      const headers = parseTableRow(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(parseTableRow(lines[index]));
        index += 1;
      }
      blocks.push(<div className="formatted-table-wrap" key={`block-${blocks.length}`}><table><thead><tr>{headers.map((cell, cellIndex) => <th key={cellIndex}>{renderInline(cell, `th-${cellIndex}`)}</th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{headers.map((_, cellIndex) => <td key={cellIndex}>{renderInline(row[cellIndex] || "", `td-${rowIndex}-${cellIndex}`)}</td>)}</tr>)}</tbody></table></div>);
      continue;
    }

    if (/^\s*>/.test(line)) {
      const quote = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) quote.push(lines[index++].replace(/^\s*>\s?/, ""));
      blocks.push(<blockquote key={`block-${blocks.length}`}>{renderInline(quote.join(" "), `quote-${blocks.length}`)}</blockquote>);
      continue;
    }

    const paragraph = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim() && !isBlockStart(lines, index)) paragraph.push(lines[index++].trim());
    blocks.push(<p key={`block-${blocks.length}`}>{renderInline(paragraph.join(" "), `paragraph-${blocks.length}`)}</p>);
  }

  return <div className="message-text formatted-message">{blocks}</div>;
}
