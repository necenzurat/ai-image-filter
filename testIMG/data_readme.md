# 🔍 Test image data

- This is a test image dataset constructed to compare functions with Google's SynthID Detector. Please unzip the dataset.zip file to use.
- The images have undergone basic editing such as bottom logo removal and format standardization (jpg), except for the dinoHash data.

## 📋 Table of Contents

- [dinoHash](#-dinoHash)
- [human](#-human)
- [view](#-view)

---
## ✨ dinoHash
- Data created by randomly selecting and re-editing AI-generated images from the [ai-vs-human-generated-dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data).
- Constructed to determine if it is possible to track images transformed by dinoHash.
- Count: 20


## ✨ human
- ID photo data of Korean men and women. Frontal photos of men and women in their 10s to 60s were generated using ChatGPT and Gemini, and then edited.
- Conducted to check if it is possible to detect images that have undergone additional editing on AI-generated portrait photos.
- Count: 24 (gpt: 12, gemini: 12)


## ✨ view
- Composed of images generated and edited by providing identical prompts to ChatGPT and Gemini based on real images and descriptions of real images. Real images and generated images were organized to the same size using Python. The prompts based on real images are as follows:

  - "A vibrant vertical street photography of Yonsei-ro, Sinchon in spring. Cherry blossom petals are falling over the pedestrian street. In the background, the iconic red U-Plex pipe structure is visible. Young students in light spring outfits are walking, blurred motion to emphasize the energy. Bright midday sunlight, crisp shadows, 8k resolution, photorealistic, urban aesthetic" (Spring scenery photo of Sinchon, Seodaemun-gu, Seoul: Key elements - falling cherry blossoms, red pipe (U-Plex), dynamism of pedestrians, bright midday light)

  - "A narrow, charming cobblestone street in Le Marais district in summer. Sunlight illuminates colorful flower boxes on window sills and vintage shop signs. A few bicycles are leaned against an old stone wall. The atmosphere is intimate and cozy, with a warm golden glow. Shot with a 50mm lens, shallow depth of field, vibrant but natural colors, 8k resolution." (Summer street scenery photo of Paris, France: Key elements - cobblestones, flower boxes by the window, bicycles, close-range density)

  - "A minimalist composition featuring a few bright orange persimmons left on a high, thin branch against a vast, clear indigo autumn sky. Minimal distractions, focusing on the silhouette and the vivid color of the fruit. The lighting is crisp and clean, emphasizing the loneliness and beauty of the season's end. Cinematic composition, high-resolution, wide-angle perspective from below." (Photo focusing on persimmons on a tree in an autumn yard: Key elements - magpie food (leftover fruit), indigo sky, minimalist composition, high clarity)
  
  - "A cinematic vertical shot of a cozy living room at night. A tall, glowing Christmas tree decorated with warm golden lights stands next to a high-end vintage wooden floor speaker. On top of the speaker, a delicate glass reindeer ornament reflects the tree's shimmer. Soft bokeh background, 8k resolution, shot on 35mm lens, moody lighting." (Winter Christmas tree photo: Key elements - golden lighting, vintage wood material, reflection of glass ornaments)

  - "High-quality upscale and detail enhancement. Aggressive denoising focused on dark shadow regions. Flawless removal of all human subjects using content-aware fill to reconstruct the background naturally." (High-quality upscale and detail enhancement. Aggressive denoising focused on dark shadow regions. Perfect subject removal using content-aware fill to naturally reconstruct the background.)

- Conducted to check if it is possible to distinguish between generated landscape photos and landscape photos where original images were additionally corrected by AI.
- Count: 18 (gpt: 6, gemini: 6, real: 6)
