"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

export default function EvaluationPage() {
  const params = useSearchParams();
  const raw = params.get("data");

  const [messages, setMessages] = useState([]);
  const [form, setForm] = useState([]);

  useEffect(() => {
    try {
      if (raw) {
        const decoded = JSON.parse(decodeURIComponent(raw));
        setMessages(decoded);

        const initial = decoded.map(() => ({
          groundTruth: "",
          context: ""
        }));
        setForm(initial);
      }
    } catch (e) {
      console.error("Invalid messages", e);
    }
  }, [raw]);

  const handleChange = (index, field, value) => {
    setForm((prev) => {
      const updated = [...prev];
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

    console.log("FINAL JSON SENT TO BACKEND:", payload);

    const res = await fetch("http://localhost:5000/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    console.log("Backend Response:", data);
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

          <div className="mt-3">
            <label className="text-sm">Context</label>
            <textarea
              className="w-full bg-black border border-gray-600 p-2 rounded-md"
              value={form[index]?.context}
              onChange={(e) => handleChange(index, "context", e.target.value)}
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
    </div>
  );
}
