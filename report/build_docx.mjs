import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  AlignmentType, BorderStyle, Document, ExternalHyperlink, Footer, HeadingLevel,
  ImageRun, Packer, PageNumber, Paragraph, Table, TableCell, TableRow,
  TextRun, WidthType,
} = require("docx");

const reportDir = path.dirname(new URL(import.meta.url).pathname);
const blocks = JSON.parse(fs.readFileSync(path.join(reportDir, "report_data.json"), "utf8"));
const ink = "26313B";
const teal = "156B78";
const muted = "4B5966";
const children = [];

function paragraph(text, options = {}) {
  return new Paragraph({
    spacing: { after: 115, line: 330 },
    ...options,
    children: [new TextRun({ text, color: ink, font: "Arial", size: 20 })],
  });
}

function caption(text) {
  return new Paragraph({
    spacing: { before: 85, after: 75 },
    keepNext: true,
    children: [new TextRun({ text, font: "Arial", size: 17, color: muted, italics: true })],
  });
}

function makeTable(rows) {
  const columns = rows[0].length;
  const widths = columns === 4 ? [1950, 4050, 1680, 1680] : [2720, 1660, 1660, 1660, 1660];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, rowIndex) => new TableRow({
      tableHeader: rowIndex === 0,
      cantSplit: true,
      children: row.map((cell, columnIndex) => new TableCell({
        width: { size: widths[columnIndex], type: WidthType.DXA },
        shading: rowIndex === 0 ? { type: "clear", fill: "E6F0F1" } : undefined,
        margins: { top: 80, bottom: 80, left: 100, right: 100 },
        children: [new Paragraph({
          spacing: { after: 0 },
          children: [new TextRun({
            text: String(cell), font: "Arial", size: 16,
            color: rowIndex === 0 ? teal : ink,
            bold: rowIndex === 0,
          })],
        })],
      })),
    })),
    borders: {
      top: { style: BorderStyle.SINGLE, color: "D6DFE1", size: 4 },
      bottom: { style: BorderStyle.SINGLE, color: "D6DFE1", size: 4 },
      left: { style: BorderStyle.SINGLE, color: "D6DFE1", size: 4 },
      right: { style: BorderStyle.SINGLE, color: "D6DFE1", size: 4 },
      insideHorizontal: { style: BorderStyle.SINGLE, color: "D6DFE1", size: 4 },
      insideVertical: { style: BorderStyle.SINGLE, color: "D6DFE1", size: 4 },
    },
  });
}

for (const block of blocks) {
  if (block.type === "title") {
    children.push(new Paragraph({
      spacing: { after: 150 },
      children: [new TextRun({ text: block.text, font: "Arial", size: 38, bold: true, color: "152238" })],
    }));
  } else if (block.type === "subtitle") {
    children.push(new Paragraph({
      spacing: { after: 180 },
      children: [new TextRun({ text: block.text, font: "Arial", size: 18, color: muted })],
    }));
  } else if (block.type === "heading") {
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_2,
      pageBreakBefore: block.text === "11. References",
      spacing: { before: 200, after: 90 },
      keepNext: true,
      children: [new TextRun({ text: block.text, font: "Arial", size: 24, bold: true, color: teal })],
    }));
  } else if (block.type === "paragraph") {
    children.push(paragraph(block.text));
  } else if (block.type === "example" || block.type === "reference") {
    children.push(new Paragraph({
      spacing: { after: 85, line: 280 },
      children: [new TextRun({ text: block.text, font: "Arial", size: 17, color: muted })],
    }));
  } else if (block.type === "link") {
    const label = new TextRun({ text: `${block.label}: `, font: "Arial", size: 17, bold: true, color: muted });
    const value = block.url
      ? new ExternalHyperlink({ link: block.url, children: [new TextRun({ text: block.url, font: "Arial", size: 17, color: teal, underline: {} })] })
      : new TextRun({ text: block.missing, font: "Arial", size: 17, color: muted });
    children.push(new Paragraph({ spacing: { after: 70 }, children: [label, value] }));
  } else if (block.type === "table") {
    children.push(caption(block.caption));
    children.push(makeTable(block.rows));
    children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  } else if (block.type === "figure") {
    const data = fs.readFileSync(path.join(reportDir, "..", block.path));
    const isCounts = path.basename(block.path) === "class_counts.png";
    children.push(new Paragraph({
      spacing: { before: 100, after: 50 },
      alignment: AlignmentType.CENTER,
      children: [new ImageRun({ data, transformation: isCounts ? { width: 450, height: 225 } : { width: 450, height: 338 } })],
    }));
    children.push(caption(block.caption));
  }
}

const document = new Document({
  creator: "Christian Tonny",
  lastModifiedBy: "Christian Tonny",
  title: "Leakage-aware Kinyarwanda news topic classification",
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1000, right: 1440, bottom: 900, left: 1440 } },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "KINNEWS  /  ", font: "Arial", size: 14, color: muted }), new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 14, color: muted })],
      })] }),
    },
    children,
  }],
});

fs.writeFileSync(path.join(reportDir, "report.docx"), await Packer.toBuffer(document));
