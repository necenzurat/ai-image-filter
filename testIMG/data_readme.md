# 🔍 Test image data

- google의 SynthID Detector와 기능 비교를 하기 위해 구축한 테스트 이미지 데이터셋입니다. dataset.zip 파일을 풀어 사용해주세요.
- 이미지는 dinoHash 데이터 외에는 하단 로고 제거/이미지 형식 변경(jpg 통일) 편집이 기본적으로 적용되어 있습니다.

## 📋 목차

- [dinoHash](#-dinoHash)
- [human](#-human)
- [view](#-view)

---
## ✨ dinoHash
- [ai-vs-human-generated-dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data)의 ai 생성 이미지를 랜덤으로 골라 재편집한 데이터입니다.
- dinoHash가 변환한 이미지 추적이 가능한지 파악하기 위해 구성하였습니다.
- 개수: 20


## ✨ human
- 한국인 남녀 증명사진 데이터입니다. 10~60 대 남녀의 정면 사진을 ChatGPT와 Gemini를 사용하여 생성 후 편집을 진행하였습니다.
- ai로 생성된 인물 사진에 추가 편집을 거친 이미지를 탐지하는 것이 가능한지 확인하려 진행하였습니다.
- 개수: 24(gpt:12, gemini:12)


## ✨ view
- 실제 이미지와 실제 이미지를 묘사한 내용을 바탕으로 프롬프트를 작성 후 ChatGPT와 Gemini에 동일하게 제공하여 이미지를 생성해 편집한 이미지로 구성되어 있습니다. 실제 이미지와 생성된 이미지는 Python을 사용하여 동일한 사이즈로 정리했습니다. 실제 이미지를 바탕으로 작성된 프롬프트는 아래와 같습니다.  

  - "A vibrant vertical street photography of Yonsei-ro, Sinchon in spring. Cherry blossom petals are falling over the pedestrian street. In the background, the iconic red U-Plex pipe structure is visible. Young students in light spring outfits are walking, blurred motion to emphasize the energy. Bright midday sunlight, crisp shadows, 8k resolution, photorealistic, urban aesthetic" (서울 서대문구 신촌의 봄 풍경 사진: 핵심 요소- 벚꽃 흩날림, 빨간 파이프(유플렉스), 보행자의 역동성, 밝은 정오의 빛)

  - "A narrow, charming cobblestone street in Le Marais district in summer. Sunlight illuminates colorful flower boxes on window sills and vintage shop signs. A few bicycles are leaned against an old stone wall. The atmosphere is intimate and cozy, with a warm golden glow. Shot with a 50mm lens, shallow depth of field, vibrant but natural colors, 8k resolution." (프랑스 파리의 여름 길거리 풍경 사진: 핵심 요소- 코블스톤(자갈길), 창가의 꽃상자, 자전거, 근거리의 밀도감)

  - "A minimalist composition featuring a few bright orange persimmons left on a high, thin branch against a vast, clear indigo autumn sky. Minimal distractions, focusing on the silhouette and the vivid color of the fruit. The lighting is crisp and clean, emphasizing the loneliness and beauty of the season's end. Cinematic composition, high-resolution, wide-angle perspective from below." (가을 마당의 나무 위의 홍시에 포커싱을 맞춰서 사진을 만들어줘. : 핵심 요소- 까치밥(남겨진 과일), 인디고 빛 하늘, 미니멀한 구도, 높은 선명도)
  
  - "A cinematic vertical shot of a cozy living room at night. A tall, glowing Christmas tree decorated with warm golden lights stands next to a high-end vintage wooden floor speaker. On top of the speaker, a delicate glass reindeer ornament reflects the tree's shimmer. Soft bokeh background, 8k resolution, shot on 35mm lens, moody lighting." (겨울 크리스마스트리 사진: 핵심 요소- 황금빛 조명, 빈티지 우드 재질, 유리 장식물의 반사광)

  - "High-quality upscale and detail enhancement. Aggressive denoising focused on dark shadow regions. Flawless removal of all human subjects using content-aware fill to reconstruct the background naturally."(고품질 업스케일 및 디테일 향상. 어두운 그림자 영역에 집중된 강력한 노이즈 제거. 내용 인식 채우기(content-aware fill)를 사용하여 배경을 자연스럽게 재구성하는 완벽한 인물 피사체 제거.)

- 생성된 풍경 사진/원본 이미지를 ai로 추가 보정한 풍경 사진 등을 판별하는 것이 가능한지 확인하려 진행하였습니다.
- 개수: 18(gpt:6, gemini:6, real:6)

