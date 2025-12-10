
"use client";
import React from 'react';
import Image from "next/image";

// import image2 from "./assets/image2.jpg";
// import item from './assets/images.png';
const ProductCard = ({ title, description, image, price }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-4 m-1">
      <Image src={image} alt={title}  className="rounded mb-4" />
      {/* <h2 className="text-lg font-bold text-gray-800 mb-2">{title}</h2> */}
      <p className="text-gray-600">{description}</p>
    </div>
  );
};

export default ProductCard;