# Recipe Goat

**Recipe Goat** is an intelligent recipe generator that creates unique, high-quality recipes based on a user-specified ingredient. Users can also customize their recipe experience by selecting a preferred **Meal** type (e.g., Breakfast, Lunch, Dinner) and **Flavor** profile (e.g., Sweet, Spicy, Bitter).

## Features

- Input a **specific food or ingredient** to generate recipes featuring it as a key component.
- Optionally select:
  - **Meal Type:** Breakfast, Lunch, Dinner, Dessert, Snack  
  - **Flavor Profile:** Salty, Sweet, Sour, Spicy, Bitter
- Intelligent multi-agent system with multiple stages:
  - Ingredient research
  - Meal and flavor application
  - Multi-recipe generation and evaluation
  - Final recipe beautification for display

---

## System Architecture

Recipe Goat is composed of five main stages, each with specialized agents that contribute to creating the final recipe:

### 1. Research Goat Stage

**Product Expert Agent:**
- Gathers detailed ingredient insights (flavor, texture, cooking styles, popularity).
- Uses **Corrective RAG** to fetch supplementary information.
  - Applies a grading system to retrieved texts.
  - If quality is poor, it reruns a refined query.
- Executes concurrent searches for:
  - `ingredient + flavor`
  - `ingredient + texture`
  - `ingredient + cooking`
- Summarizes all data with an LLM for downstream processing.

### 2. Culinary Expert Chef Goat Stage

- **Ingredient Agent:**  
  Researchs the ingredient's unique points and qualities.

- **Ingredient + Flavor Agent:**  
  Researchs the ingredient's key flavor features and flavor reviews.

- **Flavor Compatibility Agent:**  
  Researchs the ingredient's key flavors and how it would best fit a dish with the given flavor type.
  
- **Meal Agent:**  
  Researchs the ingredient's key qualities and how it might fit in a meal of the given type.

### 3. Cooking Goat Stage

- Synthesizes all prior agent outputs to create **2 complete and context-aware recipes** featuring the chosen ingredient.

### 4. Goat Board Review Stage

- Evaluates the two generated recipes.
- Selects the **best one** for final presentation using an LLM-based evaluation.

### 5. Cleaning Goat Stage

- Beautifies the final selected recipe for user-friendly display (formatting, structure, readability).

---

## Tech Stack

- **Python** (core logic, agent orchestration)
- **Threading/Concurrency** (for parallel ingredient research)
- **LLMs** (for summarization, generation, and evaluation)
- **CRAG** (Contextual Retrieval-Augmented Generation) for knowledge enrichment

---

## Example Use Case

1. **Input:**  
   - Ingredient: `Zucchini`  
   - Meal: `Dinner`  
   - Flavor: `Spicy`  

2. **Output:**  
   - One beautifully formatted spicy zucchini dinner recipe.

---

## Installation

> This section will be updated once the project is packaged for local or cloud deployment.