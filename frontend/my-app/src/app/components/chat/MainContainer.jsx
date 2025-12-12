"use client";

import React, { useState } from 'react';
import ChatArea from './ChatArea';
import RightPanel from './RightPanel';
import { FaArrowCircleUp } from "react-icons/fa";
import axios from 'axios';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import CONFIG from './config';
const MainContainer = () => {
  const [messages, setMessages] = useState([
  { input: "Hello", output: "Hi there!" },
  { input: "Summarize page 2", output: "" } // last message has empty output (loading)
]);
  const [inputmssg, setInputMssg] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [filename, setFilename] = useState('');
  const [successupload, setSuccessupload] = useState('');
  const [loading, setLoading] = useState(false); // State for loader
const handleSendtext = async () => {
    if (!inputmssg) {
      toast.error("Please enter a message first!");
      return;
    }

    try {
      setMessages((prevMessages) => [
        ...prevMessages,
        { input: inputmssg, output: '' },
      ]);

      const response = await axios.get(
        `${CONFIG.LOCALHOST}/getresult/${filename}/${inputmssg}`
      );

      console.log('Response data:', response.data.content);

      setMessages((prevMessages) =>
        prevMessages.map((msg, index) =>
          index === prevMessages.length - 1
            ? { ...msg, output: response.data.content }
            : msg
        )
      );

      setInputMssg(''); // Clear the input field
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Error sending message. Please try again.');
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setFilename('');
  };

  const handleUploadDoc = async () => {
    if (!selectedFile) {
      toast.error("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    setLoading(true); // Start the loader

    try {
      const response = await axios.post(
        `${CONFIG.LOCALHOST}/uploadpdf/`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            accept: 'application/json',
          },
        }
      );

      console.log('Upload successful:', response.data);
      setFilename(response.data.filename); // Update the filename state
      toast.success('File uploaded successfully!');
      setSuccessupload('File uploaded successfully!');
    } catch (error) {
      console.error('Error uploading file:', error);
      toast.error('Error uploading file. Please try again.');
    } finally {
      setLoading(false); // Stop the loader
    }
  };

  return (
    <div className="flex flex-col md:flex-row w-full min-h-screen">
      <ToastContainer position="top-right" autoClose={3000} />

      <div className="flex flex-col w-full md:w-3/4">
        <ChatArea messages={messages} />
        <div className="flex items-center p-2   border-t border-gray-300 rounded-lg mt-1 shadow-sm">
          <input
            type="text"
            value={inputmssg}
            onChange={(e) => setInputMssg(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 p-2 border rounded-lg placeholder-gray-400"
          />
          <button
            onClick={handleSendtext}
            className="ml-2 p-2 text-blue-500 hover:text-blue-700 "
          >
            <FaArrowCircleUp size={30} />
          </button>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            className="ml-2"
          />
          <button
            onClick={handleUploadDoc}
            className="ml-2 p-2 text-blue-500 hover:text-blue-700"
          >
            {loading ? (
              <div className="inline-block p-1.5 rounded-lg mb-2 bg-gray-300 self-start animate-pulse">
              Uploading...
            </div>
            ) : (
              'Upload File'
            )}
          </button>
        </div>
        {filename && (
          <p className="mt-2 text-sm text-green-500">
            File "{filename}" uploaded successfully!
          </p>
        )}
      </div>
      <div className="flex flex-col w-full md:w-1/4">
        <RightPanel messages={messages} />

      </div>
    </div>
  );
};


export default MainContainer;






