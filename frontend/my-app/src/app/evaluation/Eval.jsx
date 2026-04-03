"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import CONFIG from '../components/chat/config';
export default function EvaluationPage({messages}) {
   

  // const [messages, setMessages] = useState([]);
  const [form, setForm] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (Array.isArray(messages) && messages.length > 0) {
      const initial = messages.map(() => ({ groundTruth: "", context: "" }));
      setForm(initial);
    }
  }, [messages]);

  const handleChange = (index, field, value) => {
    setForm((prev) => {
      const updated = [...prev];
      if (!updated[index]) {
        updated[index] = { groundTruth: "", context: "" };
      }
      updated[index][field] = value;
      return updated;
    });
  };

  const handleSubmit = async () => {
    const payload = {
      llm_outputs: messages.map((m) => m.output),
      ground_truths: form.map((f) => f.groundTruth),
      queries: messages.map((m) => m.input),
      contexts: form.map((f) => f.context)
    };

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const res = await fetch(`${CONFIG.LOCALHOST}/getmetrics`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      console.log("Backend Response:", data);
      setResults(data.metrics || []);
    } catch (err) {
      console.error(err);
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <h1 className="text-xl font-semibold mb-4">LLM Evaluation</h1>

      {messages.map((msg, index) => (
        <div key={index} className="border border-gray-700 p-4 rounded-md mb-6">
          <h2 className="text-lg font-semibold">Query {index + 1}</h2>

          <p className="mt-2">
            <span className="font-semibold">Query:</span> {msg.input}
          </p>

          <p className="mt-1">
            <span className="font-semibold">LLM Output:</span> {msg.output}
          </p>

          <div className="mt-3">
            <label className="text-sm">Ground Truth</label>
            <textarea
              className="w-full bg-black border border-gray-600 p-2 rounded-md"
              value={form[index]?.groundTruth}
              onChange={(e) =>
                handleChange(index, "groundTruth", e.target.value)
              }
              rows={2}
            />
          </div>
 
        </div>
      ))}

      <button
        onClick={handleSubmit}
        className="w-full bg-white text-black font-bold py-2 rounded-md hover:bg-gray-300"
      >
        Submit Evaluation
      </button>
      {loading && (
        <div className="mt-4 text-center text-gray-300">Computing metrics...</div>
      )}

      {error && (
        <div className="mt-4 text-red-400">Error: {error}</div>
      )}

      {results && results.length > 0 && (
        <div className="mt-6 overflow-auto bg-white text-black rounded-md p-4">
          <h2 className="text-lg font-semibold mb-3">Evaluation Metrics</h2>
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left">
                <th className="px-2 py-1">#</th>
                <th className="px-2 py-1">Query</th> 
                <th className="px-2 py-1">Exact</th>
                <th className="px-2 py-1">F1</th>
                <th className="px-2 py-1">BLEU</th>
                <th className="px-2 py-1">ROUGE-L</th>
                <th className="px-2 py-1">Semantic Sim.</th>
                <th className="px-2 py-1">BERT Score</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i} className="border-t">
                  <td className="px-2 py-1 align-top">{i + 1}</td>
                  <td className="px-2 py-1 align-top">{r.query}</td> 
                  <td className="px-2 py-1 align-top">{r.exact_match}</td>
                  <td className="px-2 py-1 align-top">{Number(r.f1)?.toFixed(3)}</td>
                  <td className="px-2 py-1 align-top">{Number(r.bleu)?.toFixed(3)}</td>
                  <td className="px-2 py-1 align-top">{Number(r.rougeL)?.toFixed(3)}</td>
                  <td className="px-2 py-1 align-top">{Number(r.semantic_similarity)?.toFixed(3)}</td>
                  <td className="px-2 py-1 align-top">{Number(r.bert_score)?.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
