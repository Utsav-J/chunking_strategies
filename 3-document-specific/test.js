// chunking_test.js

// Example text
const text = `Chunking is the process of splitting large text into smaller, 
manageable pieces that can be stored, searched, or processed efficiently. 
It is especially important in Retrieval-Augmented Generation (RAG) 
and document indexing, since most LLMs have token limits.`;

// -------------------------------
// 1. Fixed-size chunking
// -------------------------------
function fixedSizeChunk(str, size) {
  const chunks = [];
  for (let i = 0; i < str.length; i += size) {
    chunks.push(str.slice(i, i + size));
  }
  return chunks;
}

// -------------------------------
// 2. Overlapping chunking
// -------------------------------
function overlappingChunk(str, size, overlap) {
  const chunks = [];
  for (let i = 0; i < str.length; i += (size - overlap)) {
    chunks.push(str.slice(i, i + size));
  }
  return chunks;
}

// -------------------------------
// 3. Sentence-based chunking
// -------------------------------
function sentenceChunk(str) {
  return str
    .split(/(?<=[.!?])\s+/) // split at sentence boundaries
    .map(s => s.trim())
    .filter(Boolean);
}

// -------------------------------
// Run Test
// -------------------------------
console.log("=== Fixed-size Chunks (20 chars) ===");
console.log(fixedSizeChunk(text, 20));

console.log("\n=== Overlapping Chunks (20 chars, 5 overlap) ===");
console.log(overlappingChunk(text, 20, 5));

console.log("\n=== Sentence-based Chunks ===");
console.log(sentenceChunk(text));
