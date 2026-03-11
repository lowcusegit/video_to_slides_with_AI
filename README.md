# video\_to\_slides\_with\_AI

Extracting frames from video to form a slide in my earlier repository video\_to\_slides, it is helpful for me to extract critical information for review without go thru the whole video. But it require improvement for more general use cases. That's the original.



#### Problems in earlier work

 	handcraft features only work for videos of simple slides, with no or little moving objects

 	parameters e.g. thresholds, etc. may need to adjust in different videos (minor issue)

 	these result in wrongly selecting/unselecting  frames for slides.



#### To tackle the problems
  Done (1&2)

 	possible ways 1.:revise the handcraft features and rule-base algorithm that ignore moving objects (for comparison)

 	possible ways 2.: use pretrained computer vision model for features, e.g. Image Classification (ConvNext), Text Detection, Motion Detection. Finally, Text Detection method is selected. The model DB18 or DB50 can be found from https://opencv.org/blog/text-detection-and-removal-using-opencv/#h-text-detection-models-in-opencv

  To/Not to do

 	possible ways 3.: use pretrained computer vision model for features and classification, e.g. CLIP-like models for zero-shot classification of transition slides

  Unlike to do, due to limilted resources:

 	possible ways 4.: use multimodal LLM that can "understand" the video and extract the slides (too expensive currently for long video). It is more reasonable to use LLM summarize to a list of points, instead of slides.



#### Expected outcome

 	desired slides capturing most information

