# UndergroundVisuals Rotterdam

UndergroundVisuals Rotterdam is a project for a non-profit organization organizing a citizen assembly on sustainable urban planning in Rotterdam.
The goal is to develop a web application that enables citizens to search for any address in the city and visualize both the surface view and geological layers beneath it.

![License: CC BY‑NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)

---

## Data Source
All geological data is sourced from the **Basic Registration of Subsurface Data (BRO)**, maintained by the [Dutch Ministry of Housing and Spatial Planning](https://basisregistratieondergrond.nl).

## Project Overview
The [Jupyter Notebook](dev_layer_plot.ipynb) demonstrates how to:
1. Convert a user-provided address into:
   - Geographic coordinates (latitude & longitude)
   - RD (Rijksdriehoeksmeting) coordinates
2. Retrieve and process GeoTOP subsurface data from BRO.
   - Data used focuses on South Holland, further restricted to central Rotterdam.
3. Identify the closest geological data point to the address.
4. Define a bounding box around this location.
5. Extract and visualize relevant 3D geological profiles for that area.

Example 3D visualization:
<img width="5682" height="4740" alt="3D_geological_profiles_box_for_91978_436543" src="https://github.com/user-attachments/assets/093c62d3-f0f2-4e28-8786-8692a849f81c" />

## Web Application Integration
For integration with the web platform (built with JavaScript), processed geological data is batched and exported as JSON files, each with a maximum size of 5 MB for efficient loading and display.

## Summary
This project bridges citizen participation and geospatial science, offering an interactive way to explore Rotterdam’s underground and support sustainable urban planning discussions.

---

# License

This project/repo is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
