## SketchVL: Policy Optimization via Fine-Grained Credit Assignment

## for Chart Understanding and More

## Muye Huang1,2,3, Lingling Zhang1,2*, Yifei Li1,2,3, Yaqiang Wu^4 , Jun Liu1,

(^1) School of Computer Science and Technology, Xi’an Jiaotong University, China
(^2) MOE KLINNS Lab, Xi’an Jiaotong University, China
(^3) Zhongguancun Academy, Beijing, China (^4) Lenovo Research
{huangmuye, liyifei619584902}@stu.xjtu.edu.cn
{zhanglling, liukeen}@xjtu.edu.cn, wuyqe@lenovo.com

## Abstract

```
Charts are high-density visual carriers of complex data and
medium for information extraction and analysis. Due to
the need for precise and complex visual reasoning, auto-
mated chart understanding poses a significant challenge
to existing Multimodal Large Language Models (MLLMs).
Many MLLMs trained with reinforcement learning (RL)
face the challenge of credit assignment. Their advan-
tage estimation, typically performed at the trajectory level,
cannot distinguish between correct and incorrect reason-
ing steps within a single generated response. To address
this limitation, we introduce SketchVL, a novel MLLM
that optimized with FinePO, a new RL algorithm designed
for fine-grained credit assignment within each trajectory.
SketchVL’s methodology involves drawing its intermediate
reasoning steps as markers on the image and feeding the
annotated image back to itself, creating a robust, multi-
step reasoning process. During training, the FinePO al-
gorithm leverages a Fine-grained Process Reward Model
(FinePRM) to score each drawing action within a trajec-
tory, thereby precisely assigning credit for each step. This
mechanism allows FinePO to more strongly reward cor-
rect tokens when a trajectory is globally successful, and
more heavily penalize incorrect tokens when the trajectory
is globally suboptimal, thus achieving fine-grained rein-
forcement signals. Experiments show that SketchVL learns
to align its step-level behavior with the FinePRM, achieving
an average performance gain of 7.23% over its base model
across chart datasets, natural image datasets, and math-
ematics, providing a promising new direction for training
powerful reasoning models.
```
```
*Corresponding author.
```
## 1. Introduction

```
Charts, as a primary method of data visualization, are ca-
pable of presenting data with accuracy and intuition. They
are widely found in technical documents, business reports,
and scientific papers. The automated understanding of
charts is instrumental in advancing the automated analysis
of documents, and its significant research value has progres-
sively drawn widespread attention from researchers. Recent
advanced Multimodal Large Language Models, including
Qwen2.5VL [1], Gemma3 [41], GPT-5 [35], and Gemini-
2.5 [4], have shown numerous improvements in chart un-
derstanding. Concurrently, RL in the multimodal domain,
demonstrated by approaches like Vision-R1 [15] and VLM-
R1 [38], has been shown to effectively enhance models’
multimodal understanding capabilities.
Building on these advances, chart understanding itself
exposes a distinctive, stepwise structure: models must first
localize legends and axes, then read values, align categories,
compare trends, and finally synthesize conclusions. Any
minor slip, such as an imprecise crop, a misread tick, or a
mismatched legend, can derail the entire chain of reason-
ing. This step dependency makes chart analysis particularly
sensitive to how reinforcement signals are assigned.
However, most prevailing RL practices for MLLMs still
deliver coarse, outcome-only feedback. Methods such as
GRPO [6] compute a single advantage from the final re-
sult and broadcast it uniformly to all tokens in the tra-
jectory. When applied to charts’ multi-step pipelines,
such uniform credit can inadvertently punish otherwise
sound intermediate logic in an incorrect answer, or re-
ward flawed steps simply because the final prediction hap-
pens to be right—injecting noise and limiting the bene-
fits of RL. These characteristics call for fine-grained credit
assignment: assessing and reinforcing each intermediate
step along the reasoning chain. By rewarding correct sub-
```
# arXiv:2601.05688v1 [cs.CV] 9 Jan 2026


```
Query: Around when did FDI start
consistently exceeding 400?
```
```
Deduction1: The user is asking when FDI started
exceeding 400 on the line chart. First, I should confirm the
color of the FDI curve, then locate the 400 mark, and
finally determine the corresponding time period.
Intent1: Box the FDI legend entry on the chart
to confirm the color of its corresponding curve.
Action1: #Draw Box, [y1,x1,y2,x2], Red
```
```
Deduction2: Okay, I see that the color for FDI is orange.
Now, I need to focus on the intersection point between the
orange curve and the horizontal line. This intersection
point marks the time when FDI begins to exceed 400. I
might need to draw a vertical line down from the
intersection point to determine the specific year.
Intent3: Draw a vertical line down from the
intersection point of the FDI curve and the
horizontal line to the x-axis.
Action3: #Draw Line, [y5,x5,y6,x6], Green
```
```
Intent2: Draw a horizontal line at 400.
Action2: #Draw Line, [y3,x3,y4,x4], Blue
```
```
Deduction3: Okay, I have completed all the
drawing and reasoning. We can see that the green
vertical line intersects the x-axis approximately at
```
2005. Subsequently, the orange line never crossed
back below the horizontal blue line. Evidently,
FDI has consistently remained above [the
threshold of 400] since 2005. The answer is 2005. FinePRM: [ Excellent, Acceptable ] FinePRM: [ Excellent ]

#### ... ... ... ... ... ... ... ... ... ...

```
[Adv] [Adv+delta 1 ][Adv–delta 2 ][Masked] [Adv] [Adv+delta 3 ][Masked] [Adv]
```
```
[Input Query][Input Imgae][Deduction 1 ] [Action Step 1 ] [Action Step 2 ] [Imgae] [Deduction 2 ] [Action Step 3 ] [Imgae] [Deduction 3 ]
```
```
FinePOAdvSequence:
```
Figure 1. Illustration of our SketchVL FinePO process. SketchVL first decomposes a complex query into a trajectory of visual actions,
which are scored by a FinePRM. FinePO then achieves credit assignment by redistributing credit based on the FinePRM scores. Here,
Delta 1 − 3 represent the process rewards for Actions 1-3, calculated from their FinePRM scores, and the preceding sign indicates whether
an action is considered relative advantaged or disadvantaged within the trajectory. The detailed calculation is described in Section 3.

decisions and precisely penalizing missteps, the learning
signal becomes sharper and lower-noise, better matching
the compositional nature of chart analysis.

Motivated by these observations, we introduce
SketchVL, a novel multimodal interactive reasoning
model. SketchVL operates within the Reasoning on
Image (RoI) paradigm, exemplified by works like ChartS-
ketcher [14] and DeepEyes [60]. This paradigm requires
the model to externalize its intermediate reasoning steps
into a visible trajectory of marking actions on the im-
age. This explicit, step-by-step decomposition of the
reasoning process provides the necessary structure for
step-level credit assignment. We optimize SketchVL using
our FinePO algorithm, as shown in Figure 1, which is
specifically designed to operate on these action trajectories.
FinePO first computes a relative advantage score for each
complete trajectory by making comparisons within a
group, an approach inspired by GRPO. It then introduces
its central mechanism of secondary credit redistribution.
This process leverages a specially designed Fine-grained
Process Reward Model (FinePRM) to evaluate the quality
of each individual action within the trajectory. Based on
the FinePRM scores, FinePO ultimately redistributes the

```
trajectory’s overall relative advantage among its constituent
actions according to their contributions.
To implement SketchVL, we followed the common
‘ColdStart-RL’ training paradigm. For the cold start phase,
we reused parts of the ChartSketcher data pipeline to con-
struct a dataset of 50K samples. For the subsequent rein-
forcement learning phase, we collected 9K mixed-domain
data points. Concurrently, to support the FinePO algorithm,
we built a sophisticated cross-modal distillation pipeline
to generate training data for the FinePRM. This pipeline
first decomposes dense visual information from charts into
structured textual annotations. A LLM is then used to
distill these annotations into a large-scale set of ‘intent-
action’ multimodal training pairs, complete with both pos-
itive and negative samples. In total, we collected 473K
data points through this method, specifically for training the
FinePRM’s capability to assess process quality.
Our main contributions are threefold:
```
- We introduce SketchVL, a powerful MLLM based on the
    Reasoning on Image paradigm. We also contribute two
    large-scale datasets: a 473K multimodal dataset for train-
    ing FinePRM, and a 50K dataset for RoI cold start.
- We propose FinePO, a novel reinforcement learning algo-


```
rithm that addresses the coarse reward problem in conven-
tional RL through fine-grained, step-level credit assign-
ment. By leveraging a dedicated process reward model
FinePRM, FinePO provides a more precise and stable
learning signal for policy optimization.
```
- We conduct extensive experiments across multiple bench-
    marks, demonstrating the effectiveness of FinePO. Our
    ablation studies further underscore the important role of
    credit assignment in enhancing RL performance.

## 2. Related Work

Chart Understanding. Research in Chart Understanding
focuses on interpreting the visual context of charts for tasks
such as question answering and summarization. The sem-
inal work of FigureQA [17] presented a pipeline for chart
comprehension, addressing binary classification for related
questions. Subsequent research advanced these capabilities
through the use of multi-component systems [20, 21, 25, 28,
44]. DePlot [61], for example, utilized several components
alongside the mathematical prowess of LLMs to improve
results on the PlotQA benchmark [33].
The advent of MLLMs has shifted the paradigm, lead-
ing to MLLM-centric approaches becoming the main-
stream [2, 13, 29, 30, 46, 47, 58]. By strategically con-
structing training data and fine-tuning LLaVA [23], ChartL-
lama [11] produced a capable chart-expert model. Capi-
talizing on the inherent language capabilities of MLLMs,
recent work has adopted multi-task training to enhance
chart comprehension. ChartAssistant [32], for instance, em-
ployed unified multi-task training for better overall perfor-
mance. The Program-of-Thoughts technique has been ap-
plied by TinyChart [57] to boost numerical reasoning, while
ChartMoE [52] adopted a Mixture of Experts architecture to
handle diverse chart formats.
Reasoning and RL. Models like OpenAI-o1 [34] and
Deepseek-R1 [6] have showcased the potent reasoning abil-
ities of LLMs [9, 36, 42, 48, 49, 53], which are frequently
improved via RL. Nevertheless, reasoning within MLLMs
remains a subject of active research. Many contempo-
rary methods center on employing Chain-of-Thought (CoT)
techniques [45, 56] to train MLLMs for generating step-by-
step inference sequences. Such approaches [3, 7, 24, 26]
primarily focus on CoT within the textual modality, de-
pending heavily on the MLLM’s language backbone. To
better integrate visual cues, VisualCoT [37] proposed crop-
ping critical regions to guide the model’s focus. A more
recent direction, Reasoning on Image, involves models
externalizing their reasoning by generating visual mark-
ers on the image itself, creating an interactive feedback
loop [8, 12, 22, 39, 40, 50]. More recently, attention has
also turned to the credit assignment problem in reinforce-
ment learning as a means to refine the policy optimization
process for reasoning tasks [10].

## 3. Method

```
We introduce SketchVL, a MLLM that employs an itera-
tive reasoning process. During its reasoning, the model ex-
plicitly expresses its thought process by rendering visual
markers onto the chart. This annotated chart is then fed
back to the model itself to guide the subsequent decision,
thereby forming a visible reasoning trajectory. We optimize
SketchVL using the novel FinePO, a reinforcement learn-
ing method we designed to achieve step-level, fine-grained
credit assignment. FinePO leverages our designed Fine-
grained Process Reward Model (FinePRM) to evaluate the
quality of each marking action along the reasoning path,
which in turn enables precise credit assignment. In the fol-
lowing sections, we will detail the training methodology for
SketchVL, the workflow of FinePO, and the construction
details of the FinePRM.
```
### 3.1. SketchVL

```
The training of SketchVL follows a two-stage methodol-
ogy: it first learns to generate interactive reasoning se-
quences in a Cold Start phase, followed by optimization in a
FinePO RL phase. During the cold start phase, the model is
supervised to acquire foundational localization capabilities
and robust reasoning patterns for reasoning-on-image tasks.
The subsequent RL phase then employs our FinePO algo-
rithm to unlock the model’s more complex reasoning abil-
ities. The training data for the Cold Start phase is sourced
from two main streams: the ground-truth steps generated by
the ‘Trajectorie-based Simulation’ pipeline (as will be de-
scribed in FinePRM data collection), and the ChartSketcher
data pipeline. A detailed breakdown of the data composi-
tion is provided in the Experiments section.
```
### 3.2. FinePO

```
FinePO is a reinforcement learning method that enables
step-level, fine-grained credit assignment. To achieve this,
it operates on a reasoning trajectory that is decomposed into
a sequence of discrete, visual marking actions. Each step in
this trajectory consists of a textual ‘intent’, which describes
the reasoning goal (e.g., “mark the maximum value”), and
a corresponding ‘action’, which executes this intent by ren-
dering a visual marker on the image. This mechanism of ex-
plicitly externalizing the thought process into visible steps
is fundamental to our approach.
The FinePO algorithm can be conceptualized in two
main phases. First, we compute a coarse, cross-trajectory
advantage by comparing the overall quality of different gen-
erated responses, following the approach of GRPO. Second,
we perform an intra-trajectory credit assignment, where this
coarse advantage signal is meticulously redistributed among
the fine-grained steps that constituted the response.
```

3.2.1. Cross-trajectory advantage compute
For a given prompt, we generate a set of k candidate re-
sponses, {y 1 ,y 2 ,...,yk}, which collectively form a “tra-
jectory”. Each response yiis assigned a terminal reward
R(yi) based on an evaluation of its overall correctness. The
advantage A(yi) for each response is then calculated by
subtracting the mean reward of the comparison group from
the individual response’s reward:

```
A(yi) = R(yi)−
```
#### 1

```
k
```
```
Xk
```
```
j=
```
```
R(yj) (1)
```
```
This advantage value, A(yi), indicates whether a partic-
ular response yiis better (A(yi) > 0 ) or worse (A(yi) < 0 )
than the average performance within its trajectory group.
This scalar advantage serves as the coarse, high-level signal
that will subsequently be distributed among the fine-grained
steps in the next phase.
3.2.2. Intra-trajectory credit assignment
FinePO introduces an intra-trajectory credit assignment
mechanism. This mechanism utilizes a FinePRM to assess
the quality of each individual step, allowing rewards and
penalties to be applied with greater precision to the steps
themselves, rather than being determined solely by the tra-
jectory’s final outcome. To achieve this, we first define our
FinePRM, denoted asP , as a function that provides a scalar
score for each reasoning step based on its intent and the vi-
sual change it produces:
```
```
pj=P(intentj, actionj, imgj− 1 , imgj) (2)
```
```
where pjis the process score for step sj, and imgj− 1 and
imgjare the images before and after the action is rendered.
However, the policy may develop a preference for ac-
tion types that are easier to score highly, while avoiding
actions that are important but potentially harder to execute
perfectly. This bias arises from applying a unified scoring
standard to action types that have varying levels of intrinsic
difficulty.
To counteract this bias, we introduce a KL divergence
constraint to penalize deviations between the model’s gen-
erated action distribution and a prior distribution observed
in the training set. Specifically, for a step sjwith action
type aj, we first compute a clipped KL penalty offset:
```
```
Oclipped(aj) = clip
```
#### 

```
−λKLlog
```
#### 

```
Pk(aj) + ε
Q(aj) + ε
```
#### 

```
,−γ,γ
```
#### 

#### (3)

```
where Q(a) is the pre-computed prior action distribution
from the training set, and Pk(a) is the current policy’s ac-
tion distribution computed over a sliding window of the last
k batches. λKLis the KL penalty coefficient, γ is the clip-
ping threshold, and ε ensures numerical stability.
```
```
This offset is then used to adjust the original process
score, yielding a regularized score p′j:
```
```
p′j= pj+Oclipped(aj) (4)
```
```
To perform credit assignment, we next convert these
regularized, absolute scores p′jinto values relative to the
intra-response average. This is achieved by calculating a
weighted mean. Specifically, for an entire response yi,
we weight each step’s score p′jby its corresponding token
length Lj:
```
```
̄ =p
```
#### PN

```
j=1Lj· p
```
```
′
j
PN
j=1Lj
```
#### (5)

```
The deviation for each step, ∆j= p′j− ̄, then representsp
whether a step is of higher or lower quality than the intra-
response average.
Our goal is not to create new rewards, but to redistribute
the existing coarse advantage A(yi) more accurately among
its constituent steps. This ensures that the fine-grained
learning signal remains grounded in the empirically ob-
served overall performance of the response. The weighted
sum of the adjustments across all steps is designed to be
zero, thus conserving the total advantage. We realize this
redistribution with the following formula:
```
```
A′(sj) = A(yi) + α· k· ∆j (6)
```
```
where A(yi) is the coarse advantage computed in Equa-
tion 1, α is a hyperparameter that controls the intensity of
the credit adjustment, and k is a dynamic scaling factor that
makes the adjustment’s magnitude proportional to|A(yi)|,
defined as:
```
```
k =
|A(yi)|
maxj∈{ 1 ..N}(0, ∆j) + ε
```
#### (7)

```
Here, ε is a small constant for numerical stability.
Finally, to guarantee that steps from a globally superior
response (A(yi) > 0 ) do not receive negative advantages,
and vice versa, we apply a clipping function to produce the
final step-level advantage A(sj):
```
```
A(sj) =
```
#### (

```
clip(A′(sj), 0 ,β· A(yi)) if A(yi) > 0
clip(A′(sj),β· A(yi), 0) if A(yi)≤ 0
```
#### (8)

```
where β is a hyperparameter that defines the bounds of the
clipping range. This final step ensures a stable signal for
each individual action in the policy optimization process.
```
### 3.3. FinePRM

```
The FinePRM is a process reward model that provides
the signal for FinePO’s fine-grained credit assignment. In
the following sections, we detail the architecture of the
FinePRM and the methodology for its data collection.
```

```
SAM/PaddleOCR
Decomposition SelfInteractive attributes-attributes: This is the chart title: “Positive economic...since 2016.: The chart title is placed at the top ...
BBOX: [ y1,x1,y2,x2 ]
```
```
SelfInteractive attributes-attributes: This is the legend, indicating the Bad category in blue.: Legend positioned near the top; title above...
BBOX: [ y1,x1,y2,x2 ]
```
```
SelfInteractive attributes-attributes: The X-: Xaxis label “2016” denotes the year 2016..-axis labels: 2014 (left) to 2018 (right) ...
BBOX: [ y1,x1,y2,x2 ]
```
```
SelfInteractive attributes-attributes: The number 59, in blue.: Above the blue line, with 71 on the left ...
BBOX: [ y1,x1,y2,x2 ]
```
```
Intent: Circle the “Bad” legend in the
chart.
Action: Draw Box, [y1,x1,y2,x2], Red
```
```
QwenLLM
```
```
Annotation
```
```
Intent Distillation
```
```
Intent: Circle the “Bad” legend in the
chart.
Action: Draw Box, [y1,x1,y2,x2], Red
```
```
Noise Injection
```
```
Visual-to-Text Annotation Text-to-Image Distillation
```
```
Segments
```
Figure 2. The data pipeline for training our FinePRM. First, we convert charts into structured textual annotations via decomposition and
description (left). Second, we use LLMs to distill these annotations into a training dataset of ‘intent-action’ pairs, including both positive
samples and negative samples generated via noise injection (right).

3.3.1. Architecture of FinePRM

Similar to prior work VisualPRM [43], we employ a MLLM
as the backbone of our FinePRM. As defined previously, the
FinePRM takes as input the textual ‘intent’ and ‘action’,
alongside the pair of images before and after the action,
imgj− 1 and imgj.
These inputs are structured using a specific prompt tem-
plate that frames the task as an evaluative judgment. The
MLLM is presented with both images and is guided by a
textual query to act as an assessor. The query instructs the
model to carefully compare the visual modification between
the two images and evaluate whether this change is a pre-
cise and correct execution of the given ‘intent’ and ‘action’.
A conceptual representation of the query is as follows:

```
Image Before: [Image 1]
Image After: [Image 2]
The following modification was ... Critically
evaluate if the modification in Image 2 is
a precise and correct realization of this
intent. Classify the quality into one of
four levels: [Excellent, Acceptable, Poor,
Unacceptable].
```
We formulate this as a four-way classification task. The
FinePRM is trained to output one of the four discrete labels:
Excellent, Acceptable, Poor, or Unacceptable. These cate-
gorical judgments are subsequently mapped to scalar values
[4. 0 , 3. 0 , 2. 0 , 1 .0] to serve as the process score p′jrequired
for the intra-trajectory credit assignment phase.

3.3.2. FinePRM Training Data Collection

Constructing the training data for FinePRM requires a
large-scale dataset of ‘intent-action’ pairs. However, even
advanced MLLMs, such as Gemini 2.5 Pro, still struggle
with complex visual grounding. Consequently, conven-
tional distillation approaches are both cost-prohibitive and
incapable of generating ‘intent-action’ pairs with the re-
quired precision. To address this issue, we employ a so-

```
phisticated cross-modal distillation method that transforms
raw images into precise ‘intent-action’ annotations. Our ap-
proach is executed in two main stages. First, we translate
the visual modality of an image into a dense, intermedi-
ate representation composed of [text description & target
location] annotation pairs. Subsequently, these structured
annotations are processed by LLMs to distill the final mul-
timodal training data for FinePRM.
```
1. Visual-to-Text Annotation MLLMs struggle to simul-
taneously localize and identify a large number of objects in
dense images. To address this, we require precise grounding
information for each object. Inspired by Set-of-Mark [55],
we leverage the SAM [18] as an auxiliary segmentation tool
to provide this information. We first employ SAM to parti-
tion an image into numerous object-centric patches. These
patches are then fed into a leading open-source MLLM for
labeling. To provide local context, we expand each patch to
include an additional 20% surrounding area from the origi-
nal image. We then center the target object within this ex-
panded view and highlight it with a red bounding box before
presenting it to the MLLM for annotation.
    For each processed patch, we prompt the MLLM to gen-
erate two types of attributes for the highlighted object:
- Self-attributes: The object’s intrinsic name or properties,
    such as “lid of a starfruit can” or “a purple polyline”.
- Interactive attributes: The object’s relationship with its
    surroundings, such as “a jar of peanut butter is to its left”
    or “intersects with the green polyline”.
       As shown in Figure 2, this approach significantly reduces
the model’s cognitive load, leveraging the MLLM’s strength
in single-object recognition to effectively convert the visual
content into a rich set of textual annotations. It is worth
noting that we employ PaddleOCR for text recognition, as


SAM is not adept at this task.

2. Text-to-Image Distillation In this stage, we use
the structured annotations generated previously to prompt
LLM, distilling them into a comprehensive dataset of simu-
lated ‘intent-action’ pairs. For each generated pair, we em-
ploy a rendering engine to apply the ‘action’ to the origi-
nal image, thereby producing the imgj− 1 (before) and imgj
(after) states required for FinePRM’s input. To ensure the
diversity of the training data, we synthesize these ‘intent-
action’ pairs through two distinct pipelines:
- Direct Generation: The LLM is prompted to directly
    generate a straightforward ‘intent-action‘ pair from the
    annotations. This typically results in simple, single-step
    tasks, such as an intent to “Mark the man with the white
    beard” or “Highlight the 50% value label on the x-axis”.
- Trajectory-based Simulation: This pipeline is more
    complex and designed to mimic the reasoning process of
    FinePO. We first use the annotations to create a question-
    answering pair. A powerful LLM is then tasked to answer
    the question, simulating the generation of ‘trajectories‘ as
    seen in FinePO. This process yields multi-step, ground-
    truth (GT) reasoning trajectories, and we harvest these
    individual steps for subsequent training.
       Finally, we inject noise to the ‘action’ component of
the collected GT steps. This process systematically gen-
erates samples exhibiting a mismatch between the stated
‘intent’ and the executed ‘action’. The final training data
for FinePRM is compiled with a specific 2:4:3:1 ratio for
the Excellent, Acceptable, Poor, and Unacceptable labels,
respectively. This non-uniform distribution is deliberately
chosen to compel the model to focus on the critical and of-
ten subtle decision boundary between the ‘Acceptable’ and
‘Poor’ categories.

## 4. Experiments

### 4.1. Settings

4.1.1. Data Construction

Cold Start Data. For the cold-start phase, we selected im-
ages from the EvoChart, GQA, and ChartQA-Train datasets
as metadata. For the EvoChart dataset, which contains
synthesis source code, we reused the ChartSketcher data
pipeline. For GQA and ChartQA-Train, we utilized the
Visual-to-Text annotations generated during our FinePRM
construction process. It is noteworthy that all QA pairs were
synthetically generated; we did not use any QA pairs from
the original training sets of these datasets. The final SFT
data, totaling 50K samples, was distilled using the Qwen3-
235B-A22B-Instruct-2507 [54].
FinePRM Training Data. For the construction of the
FinePRM, we used images from EvoChart [13], GQA [16],
ChartQA-Train [27], and OpenImages [19] as data sources.

```
The data pipeline employed SAM [18] as a segmentation
tool, PaddleOCR V5 [5] as an OCR tool, Qwen2.5VL-
72B-Instruct [1] for Visual-to-Text Annotation, and Qwen-
NEXT-80B-A3B for Text-to-Image Distillation. We defined
five action types: Line, Point, Rectangle, Circle,
and Text. The Text action is excluded from credit as-
signment, and the data for the remaining four actions is
balanced with a 1:1:1:1 ratio. The distribution for the Ex-
cellent, Acceptable, Poor, and Unacceptable labels across
all data was set to a 2:4:3:1 ratio. In total, we collected
473K SFT samples to train the FinePRM’s capability to as-
sess process quality.
RL Data. In the FinePO RL phase, we aggregated
9k prompts from five distinct data sources for online tra-
jectory generation. The composition is as follows: 2k
from ChartQA-Train-Augmented, 2k from ChartQA-Train-
Human, 1k from Vision-R1-RL, 2k from ChartBench-
Train [51] , and 2k from VisualCoT-Train.
4.1.2. Evaluation
Benchmarks: We conducted extensive experiments on
both Chart Expert and General-Purpose datasets.
```
- Chart Expert Datasets: EvoChart-QA, ChartQA,
    ChartQA-Pro [31] , ChartBench, and PlotQA [33].
- General-Purpose Datasets: MMStar and MathVista.
    Evaluation Protocol: We employ DeepSeek-R1-Distill-
Qwen-14B as an evaluation discriminator. The evaluation
rules for each dataset are provided as a prompt, and the cor-
rectness of a response is determined by a 9-vote majority.
4.1.3. Training Settings
We trained two versions of SketchVL, based on
Qwen2.5VL-7B-Instruct and Qwen2.5VL-3B-Instruct,
respectively. The FinePRM was trained using Qwen2.5VL-
7B-Instruct as its backbone. All experiments were
conducted using the ms-swift v3.9.0.dev0 [59] framework.
The key hyperparameters for the FinePO reinforcement
learning phase are detailed in the Table 1. FinePRM
training for 4 epochs, cold start for 2 epochs and FinePO
training for 1 epochs. All models were trained on 16 x
NVIDIA A800 (40G) GPUs.

```
Table 1. Key hyperparameters for the FinePO RL phase.
```
```
Symbol Meaning Value
k Number of generations per prompt 24
λKL KL penalty coefficient 0.
γ Clipping threshold for KL offset 0.
α Credit adjustment intensity 0.
β Clipping range factor for advantage 2.
```
- Learning Rate 1e-
- Temperature 1.
- GRPO KL beta 0.


```
Table 2. Performance comparison of SketchVL with other leading models and results of our ablation study across a selection of chart-expert
and general-purpose benchmarks. *Random means that, during training, random rewards are used in place of the FinePRM.
```
Model EvoChart-QA ChartQA ChartQA-Pro ChartBench PlotQA MathVista MMStar
Performance Comparison
VLM-R1 40.32 72.98 39.58 39.58 54.40 55.10 48.
ChartSketcher-2B 26.72 68.24 - 30.10 41.12 - -
Qwen2.5VL-7B 54.80 82.00 52.40 64.78 63.44 61.40 56.
Qwen2.5VL-3B 39.36 61.88 37.73 56.20 42.88 49.50 43.
SketchVL-7B (Ours) 58.64 83.96 52.62 65.11 55.84 63.50 57.
SketchVL-3B (Ours) 47.28 77.20 44.15 59.96 48.32 53.80 51.
Ablation Study (on SketchVL-3B)
SketchVL-3B (Full Model) 47.28 77.20 44.15 59.96 48.32 53.90 52.
w/o FinePO (only cold start) 42.08 71.80 38.45 54.57 35.44 47.60 49.
w/o FinePO (naive GRPO) 45.60 75.12 43.69 59.26 44.72 53.80 49.
w/o FinePRM (random* scores) 48.08 76.76 43.94 58.98 46.40 52.30 50.
w/o KL Action Regularization 48.56 77.80 43.33 58.24 48.16 54.60 51.
w/o Sketch (zero GRPO) 30.48 57.56 25.92 53.06 31.12 45.40 43.
w/o RL (use SFT) 26.48 54.72 20.02 50.24 27.44 43.90 38.

### 4.2. Performance Comparison

```
As shown in Table 2, SketchVL demonstrates promising
performance across multiple benchmarks.
First, our models significantly outperform their base
models. SketchVL-7B surpasses Qwen2.5VL-7B on al-
most all chart-expert datasets, validating the effectiveness
of the FinePO training method. This performance lift is
even more pronounced on the 3B-scale model; SketchVL-
3B achieves substantial gains over its base, Qwen2.5VL-
3B, outperforming it by 15.32 and 3.76 percentage points
on tasks like ChartQA and ChartBench, respectively. It is
noteworthy that the magnitude of this improvement is more
pronounced for the 3B model than for the 7B. We hypoth-
esize this is because the smaller model has less capacity
to “hack” the large-scale FinePRM, compelling it to more
faithfully learn the intended reasoning process from signals.
Second, SketchVL also performs exceptionally well
when compared against reasoning models of a similar scale.
For instance, our SketchVL-3B substantially outperforms
VLM-R1 across most listed benchmarks. When com-
pared to ChartSketcher, another leading interactive reason-
ing model, our model significantly surpasses its 2B version.
Finally, it is noteworthy that SketchVL maintains strong
general-purpose multimodal capabilities while enhancing
its chart understanding skills. On MathVista and MMStar,
two non-chart-centric datasets, SketchVL-7B achieves re-
sults that are competitive with or superior to leading models.
This indicates that FinePO not only boosts performance on
specialized tasks but also successfully preserves the model’s
generalization abilities.
```
### 4.3. Ablation Study

```
To investigate the individual contributions of FinePO’s core
components, we conducted a series of ablation experiments
using the smaller SketchVL-3B model. This allows for ef-
ficient verification of our methodological design choices.
Value of Fine-Grained Credit Assignment in FinePO.
The core of our experiments lies in validating the pro-
posed fine-grained credit assignment mechanism. Com-
pared to the baseline using naive GRPO (‘w/o FinePO
(naive GRPO)’), our full model achieves consistent per-
formance gains across several key benchmarks, such as
a 2.08-point improvement on ChartQA. This advantage is
further substantiated by the performance drop in the ‘w/o
FinePRM (random scores)’ experiment, which confirms
that our trained FinePRM provides a meaningful and neces-
sary evaluation signal, rather than random noise. Together,
these points demonstrate the value of FinePO.
Role of KL Action Regularization. The ‘w/o KL ac-
tion Regularization’ model achieves performance very close
to our full model. This is an expected outcome, as the KL
constraint introduces a trade-off between action precision
and diversity. It effectively prevents the model from col-
lapsing to a few easy-to-score action types during training,
but it can sometimes slightly constrain the optimal policy on
benchmarks where a specific action type is dominant. We
provide detailed analysis of this trade-off in Section 4.4.
Impact of RL and Generalization. As a whole, the
RL phase provides a substantial performance boost. Com-
pared to the model after only the cold-start phase, our full
model shows a significant leap in performance. Notably,
this improvement also extends to datasets not covered in the
```

```
Query: What is the value of bad in the year 2016?
Step1 - 1 : Draw a rectangle on the chart, highlighting the
'Bad' data label for 2016 to clearly indicate its position.
Step1-1 Score: Acceptable
Step1- 2 : Draw a line connecting this label to the blank area
on the right, to prepare for subsequent value annotation.
Step 1 - 2 : Score: Acceptable
Step2 - 1 : Draw a rectangle to highlight the numerical part
of this data label, emphasizing its content.
Step2-1 Score: Excellent
Step2 - 2 : Draw a text annotation, indicating the value is 71.
FinalAnswer: Confirm through annotation that in 2016, the
value for the 'Bad' category is 71. This value should be
directly annotated on the corresponding data point.
```
```
Intent:
“Mark the year 2016 on
the x-axis with a point, in
preparation for the next
step of reasoning.”
Excellent
Acceptable
Poor
Unacceptable
```
```
SketchVL Interactive Reasoning & FinePRM Scoring FinePRM Scoring Heatmap
```
Figure 3. Left: A successful case of SketchVL’s interactive reasoning, with each step scored by the FinePRM. Right: The corresponding
scoring heatmap generated by the FinePRM for a specific intent.

(^0) line rectangle circle point text
20
40
60
80
Percentage (%)
SketchVL-7B
line rectangle circle point text
SketchVL-3B
Action Type Distribution
Cold Start w/o KL Regularization Full Model (w/ KL)
Figure 4. Effect of KL regularization on the action type distribu-
tion for SketchVL-7B (left) and SketchVL-3B (right).
RL training prompts, such as PlotQA. This indicates that
FinePO can enhance generalization capabilities.
Importance of the RoI Paradigm. The results from the
‘w/o Sketch (zero GRPO)’ experiment highlight the foun-
dational role of the RoI paradigm. Removing the model’s
ability to draw on the image leads to a drastic collapse in
performance across all benchmarks. This confirms that the
iterative, visual reasoning process is fundamental for solv-
ing these complex tasks.

### 4.4. Analysis of Reasoning Process

To evaluate the model’s quality beyond final task accuracy,
we further analyze its intermediate reasoning process using
our FinePRM as an automatic evaluator.
As presented in Table 3, our full models consistently
achieve the highest average process scores across all test
sets. This result clearly demonstrates that FinePO’s fine-
grained signal effectively aligns the model with the PRM.
Notably, this superior process quality is demonstrated on
datasets such as PlotQA, which were not included in the RL
training phase, proving that FinePO enhances the model’s
intrinsic and generalizable reasoning capabilities.

```
The importance of KL action regularization is visually
demonstrated in Figure 4. While the model trained with-
out the KL constraint improves upon the cold start model
in process scores, it exhibits a severe “Action Bias”, heavily
favoring a few action types. This action bias is particularly
severe in the 7B model, whose action distribution almost
completely collapses, likely due to its larger model capacity
providing more opportunities to discover and exploit poten-
tial shortcuts or loopholes within the FinePRM. Therefore,
the KL constraint does not introduce a performance trade-
off in this case; instead, it prevents the model from adopting
a simplistic, biased policy. By encouraging a more diverse
and balanced use of actions, it compels the model to develop
a more robust reasoning strategy, which ultimately leads to
higher-quality intermediate steps.
```
```
Table 3. Average FinePRM scores on test sets for models in the
ablation study. Higher scores indicate better alignment between
generated actions and their intents. *Random means that, during
training, random rewards are used in place of the FinePRM.
```
```
Model PlotQA ChartBench MMStar
SketchVL-3B (Full) 2.857 2.917 2.
w/o FinePO (use GRPO) 2.705 2.777 2.
w/o FinePRM (random*) 2.631 2.618 2.
w/o KL Regularization 2.825 2.818 2.
w/o RL (only cold start) 2.747 2.836 2.
SketchVL-7B (Full) 3.003 3.073 2.
w/o KL Regularization 2.908 2.921 2.
w/o RL (only cold start) 2.828 2.897 2.
```
### 4.5. Case Visualization

```
Figure 3 provides a qualitative visualization of SketchVL’s
reasoning process. The left panel illustrates a successful
case where each of SketchVL’s reasoning steps is scored by
our FinePRM. Due to the absence of a standardized bench-
mark for evaluating process reward models, we assess the
```

FinePRM’s scoring capability intuitively through visualiza-
tion. To produce the scoring heatmap, we define an in-
tent, apply the action across an 8×8 image grid, and have
FinePRM score each action. The resulting heatmap demon-
strates that our FinePRM can clearly distinguish between
correct and incorrect regions, assigning the highest scores
to the areas that most accurately fulfill the intent.

## 5. Conclusion and Limitations

In this work, we present SketchVL, a multimodal reasoning
model trained with our RL algorithm, FinePO. By lever-
aging a FinePRM, FinePO achieves step-level credit as-
signment, alleviating the limitations of coarse reward sig-
nals in chart reasoning tasks. Our experiments demon-
strate SketchVL’s strong performance across diverse chart
understanding and multimodal benchmarks. We also ac-
knowledge the limitations of our work, which point to di-
rections for future research. These include the current
absence of a benchmark for quantitatively evaluating the
FinePRM and the inherent challenge of constructing a un-
biased PRM.

## References

```
[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, and Peng Wang. Qwen2.5-vl tech-
nical report, 2025. 1, 6
[2] Jinyue Chen, Lingyu Kong, Haoran Wei, Chenglong Liu,
Zheng Ge, Liang Zhao, Jianjian Sun, Chunrui Han, and
Xiangyu Zhang. Onechart: Purify the chart struc-
tural extraction via one auxiliary token. arXiv preprint
arXiv:2404.09987, 2024. 3
[3] Kanzhi Cheng, Wenpo Song, Jiaxin Fan, Zheng Ma, Qiushi
Sun, Fangzhi Xu, Chenyang Yan, Nuo Chen, Jianbing
Zhang, and Jiajun Chen. Caparena: Benchmarking and ana-
lyzing detailed image captioning in the LLM era. In Findings
of the Association for Computational Linguistics, ACL 2025,
Vienna, Austria, July 27 - August 1, 2025, pages 14077–
```
14094. Association for Computational Linguistics, 2025. 3
[4] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice
Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blis-
tein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris,
Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz,
Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-
Jiang Jiang, and Krishna Haridasan. Gemini 2.5: Pushing the
frontier with advanced reasoning, multimodality, long con-
text, and next generation agentic capabilities, 2025. 1
[5] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo
Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda
Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang,
Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu,
and Yanjun Ma. Paddleocr 3.0 technical report, 2025. 6
[6] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai
Yu, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning
capability in llms via reinforcement learning, 2025. 1, 3

```
[7] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang,
Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Ex-
ploring long-chain visual reasoning with multimodal large
language models. CoRR, abs/2411.14432, 2024. 3
[8] Xingyu Fu, Minqian Liu, Zhengyuan Yang, John Corring,
Yijuan Lu, Jianwei Yang, Dan Roth, Dinei A. F. Florˆencio,
and Cha Zhang. Refocus: Visual editing as a chain of thought
for structured image understanding. CoRR, abs/2501.05452,
```
2025. 3
[9] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Ab-
hinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy
Yang, Angela Fan, Zhenyu Yang, Zhiwei Zhao, and Zhiyu
Ma. The llama 3 herd of models, 2024. 3
[10] Yiran Guo, Lijie Xu, Jie Liu, Dan Ye, and Shuang Qiu. Seg-
ment policy optimization: Effective segment-level credit as-
signment in rl for large language models, 2025. 3
[11] Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang,
Gang Yu, Bin Fu, and Hanwang Zhang. Chartllama: A mul-
timodal LLM for chart understanding and generation. arXiv
preprint arXiv:2311.16483, 2023. 3
[12] Jiaqi Huang, Zunnan Xu, Jun Zhou, Ting Liu, Yicheng
Xiao, Mingwen Ou, Bowen Ji, Xiu Li, and Kehong Yuan.
SAM-R1: leveraging SAM for reward feedback in mul-
timodal segmentation via reinforcement learning. CoRR,
abs/2505.22596, 2025. 3
[13] Muye Huang, Han Lai, Xinyu Zhang, Wenjun Wu, Jie Ma,
Lingling Zhang, and Jun Liu. Evochart: A benchmark and a
self-training approach towards real-world chart understand-
ing. In AAAI-25, Sponsored by the Association for the Ad-
vancement of Artificial Intelligence, February 25 - March
4, 2025, Philadelphia, PA, USA, pages 3680–3688. AAAI
Press, 2025. 3, 6
[14] Muye Huang, Lingling Zhang, Jie Ma, Han Lai, Fangzhi Xu,
Yifei Li, Wenjun Wu, Yaqiang Wu, and Jun Liu. Charts-
ketcher: Reasoning with multimodal feedback and reflection
for chart understanding. CoRR, abs/2505.19076, 2025. 2
[15] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao,
Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin.
Vision-r1: Incentivizing reasoning capability in multimodal
large language models. CoRR, abs/2503.06749, 2025. 1
[16] Drew A. Hudson and Christopher D. Manning. GQA: A new
dataset for real-world visual reasoning and compositional
question answering. In IEEE Conference on Computer Vi-
sion and Pattern Recognition, CVPR 2019, Long Beach, CA,
USA, June 16-20, 2019, pages 6700–6709. Computer Vision
Foundation / IEEE, 2019. 6
[17] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkin-
son,Akos K ́ ad ́ar, Adam Trischler, and Yoshua Bengio. Fig- ́
ureqa: An annotated figure dataset for visual reasoning. In
ICLR, 2018. 3
[18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C. Berg, Wan-Yen Lo, Piotr Doll ́ar, and
Ross Girshick. Segment anything, 2023. 5, 6
[19] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper R. R.
Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Ste-
fan Popov, Matteo Malloci, Tom Duerig, and Vittorio Ferrari.


The open images dataset V4: unified image classification,
object detection, and visual relationship detection at scale.
CoRR, abs/1811.00982, 2018. 6
[20] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu,
Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandel-
wal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova.
Pix2struct: Screenshot parsing as pretraining for visual lan-
guage understanding. In ICML, pages 202: 18893–18912,

2023. 3
[21] Matan Levy, Rami Ben-Ari, and Dani Lischinski.
Classification-regression for chart comprehension. In
ECCV, pages 13696: 469–484, 2022. 3
[22] Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia,
Shaoguang Mao, Li Dong, Ivan Vulic, and Furu Wei. Imag-
ine while reasoning in space: Multimodal visualization-of-
thought. CoRR, abs/2501.07542, 2025. 3
[23] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li,
Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave:
Tackling multi-image, video, and 3d in large multimodal
models. arXiv preprint arXiv:2407.07895, 2024. 3
[24] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang,
Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Mon-
key: Image resolution and text label are important things
for large multi-modal models. In IEEE/CVF Conference
on Computer Vision and Pattern Recognition, CVPR 2024,
Seattle, WA, USA, June 16-22, 2024, pages 26753–26763.
IEEE, 2024. 3
[25] Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi
Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Col-
lier, and Julian Martin Eisenschlos. Matcha: Enhancing
visual language pretraining with math reasoning and chart
derendering. In ACL, pages 12756–12770, 2023. 3
[26] Zuyan Liu, Yuhao Dong, Yongming Rao, Jie Zhou, and Ji-
wen Lu. Chain-of-spot: Interactive reasoning improves large
vision-language models. CoRR, abs/2403.12966, 2024. 3
[27] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty,
and Enamul Hoque. Chartqa: A benchmark for question an-
swering about charts with visual and logical reasoning. In
Findings of ACL, pages 2263–2279, 2022. 6
[28] Ahmed Masry, Parsa Kavehzadeh, Do Xuan Long, Ena-
mul Hoque, and Shafiq Joty. Unichart: A universal vision-
language pretrained model for chart comprehension and rea-
soning. In EMNLP, pages 14662–14684, 2023. 3
[29] Ahmed Masry, Mehrad Shahmohammadi, Md. Rizwan
Parvez, Enamul Hoque, and Shafiq Joty. Chartinstruct:
Instruction tuning for chart comprehension and reasoning.
arXiv preprint arXiv:2403.09028, 2024. 3
[30] Ahmed Masry, Megh Thakkar, Aayush Bajaj, Aaryaman
Kartha, Enamul Hoque, and Shafiq Joty. Chartgemma: Vi-
sual instruction-tuning for chart reasoning in the wild, 2024.
3
[31] Ahmed Masry, Mohammed Saidul Islam, Mahir Ahmed,
Aayush Bajaj, Firoz Kabir, Aaryaman Kartha, Md Tah-
mid Rahman Laskar, Mizanur Rahman, Shadikur Rah-
man, Mehrad Shahmohammadi, Megh Thakkar, Md Rizwan
Parvez, Enamul Hoque, and Shafiq Joty. Chartqapro: A
more diverse and challenging benchmark for chart question
answering, 2025. 6

```
[32] Fanqing Meng, Wenqi Shao, Quanfeng Lu, Peng Gao,
Kaipeng Zhang, Yu Qiao, and Ping Luo. Chartassisstant: A
universal chart multimodal language model via chart-to-table
pre-training and multitask instruction tuning. arXiv preprint
arXiv: 2401.02384, 2024. 3
[33] Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and
Pratyush Kumar. Plotqa: Reasoning over scientific plots. In
WACV, pages 1516–1525, 2020. 3, 6
[34] OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam
Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar,
Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie,
Alex Karpenko, and Alex Tachard Passos Zhuohan Li. Ope-
nai o1 system card, 2024. 3
[35] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, and Florencia Leoni Aleman et
al. Gpt-4 technical report, 2024. 1
[36] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan
Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, and Zihan Qiu. Qwen2.5 technical report, 2025.
3
[37] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan
Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot:
Advancing multi-modal language models with a comprehen-
sive dataset and benchmark for chain-of-thought reasoning.
In Advances in Neural Information Processing Systems 38:
Annual Conference on Neural Information Processing Sys-
tems 2024, NeurIPS 2024, Vancouver, BC, Canada, Decem-
ber 10 - 15, 2024, 2024. 3
[38] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo
Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao,
Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. VLM-
R1: A stable and generalizable r1-style large vision-language
model. CoRR, abs/2504.07615, 2025. 1
[39] Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and
Wenhu Chen. Pixel reasoner: Incentivizing pixel-space rea-
soning with curiosity-driven reinforcement learning. CoRR,
abs/2505.15966, 2025. 3
[40] Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao,
Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Jun-
tao Li, Xiaoye Qu, and Yu Cheng. Openthinkimg: Learning
to think with images via visual tool reinforcement learning.
CoRR, abs/2505.08617, 2025. 3
[41] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya
Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Ta-
tiana Matejovicova, Alexandre Ram ́e, Morgane Riviere,`
Louis Rouillard, and Thomas Mesnard. Gemma 3 technical
report, 2025. 1
[42] Qwen Team. Qwq-32b: Embracing the power of reinforce-
ment learning, 2025. 3
[43] Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jin-
guo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Sheng-
long Ye, Xizhou Zhu, Lewei Lu, Haodong Duan, Yu Qiao,
Jifeng Dai, and Wenhai Wang. Visualprm: An effective
process reward model for multimodal reasoning. CoRR,
abs/2503.10291, 2025. 5
[44] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao
Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sad-
hika Malladi, Alexis Chevalier, Sanjeev Arora, and Danqi
```

```
Chen. Charxiv: Charting gaps in realistic chart understand-
ing in multimodal llms. arXiv preprint arXiv:2406.18521,
```
2024. 3
[45] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and
Denny Zhou. Chain-of-thought prompting elicits reasoning
in large language models. In Advances in Neural Informa-
tion Processing Systems 35: Annual Conference on Neural
Information Processing Systems 2022, NeurIPS 2022, New
Orleans, LA, USA, November 28 - December 9, 2022, 2022.
3
[46] Renqiu Xia, Bo Zhang, Haoyang Peng, Ning Liao, Peng Ye,
Botian Shi, Junchi Yan, and Yu Qiao. Structchart: Percep-
tion, structuring, reasoning for visual chart understanding.
CoRR, abs/2309.11268, 2023. 3
[47] Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi
Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi,
Junchi Yan, and Yu Qiao. Chartx & chartvlm: A versatile
benchmark and foundation model for complicated chart rea-
soning. CoRR, abs/2402.12185, 2024. 3
[48] Fangzhi Xu, Qiushi Sun, Kanzhi Cheng, Jun Liu, Yu Qiao,
and Zhiyong Wu. Interactive evolution: A neural-symbolic
self-training framework for large language models. In Pro-
ceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), ACL
2025, Vienna, Austria, July 27 - August 1, 2025, pages
12975–12993. Association for Computational Linguistics,
2025. 3
[49] Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi
Sun, Kanzhi Cheng, Junxian He, Jun Liu, and Zhiyong
Wu. Genius: A generalizable and purely unsupervised self-
training framework for advanced reasoning. In Proceedings
of the 63rd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), ACL 2025, Vi-
enna, Austria, July 27 - August 1, 2025, pages 13153–13167.
Association for Computational Linguistics, 2025. 3
[50] Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang,
Anna Korhonen, and Ivan Vulic. Visual planning: Let’s think
only with images. CoRR, abs/2505.11409, 2025. 3
[51] Zhengzhuo Xu, Sinan Du, Yiyan Qi, Chengjin Xu, Chun
Yuan, and Jian Guo. Chartbench: A benchmark for complex
visual reasoning in charts. CoRR, abs/2312.15915, 2023. 6
[52] Zhengzhuo Xu, Bowen Qu, Yiyan Qi, Sinan Du, Chengjin
Xu, Chun Yuan, and Jian Guo. Chartmoe: Mixture of di-
versely aligned expert connector for chart understanding,
2025. 3
[53] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen
Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng
Liu, and Fei Huang. Qwen2 technical report. arXiv preprint
arXiv:2407.10671, 2024. 3
[54] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou,
and Fei Huang. Qwen3 technical report, 2025. 6
[55] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chun-
yuan Li, and Jianfeng Gao. Set-of-mark prompting un-
leashes extraordinary visual grounding in GPT-4V. CoRR,
abs/2310.11441, 2023. 5

```
[56] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom
Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of
thoughts: Deliberate problem solving with large language
models. In Advances in Neural Information Processing Sys-
tems 36: Annual Conference on Neural Information Process-
ing Systems 2023, NeurIPS 2023, New Orleans, LA, USA,
December 10 - 16, 2023, 2023. 3
[57] Liang Zhang, Anwen Hu, Haiyang Xu, Ming Yan, Yichen
Xu, Qin Jin, Ji Zhang, and Fei Huang. Tinychart: Efficient
chart understanding with visual token merging and program-
of-thoughts learning. arXiv preprint arXiv: 2404.16635,
```
2024. 3
[58] Shuoshuo Zhang, Zijian Li, Yizhen Zhang, Jingjing Fu, Lei
Song, Jiang Bian, Jun Zhang, Yujiu Yang, and Rui Wang.
Pixelcraft: A multi-agent system for high-fidelity visual rea-
soning on structured images, 2025. 3
[59] Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yun-
lin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai,
Ang Wang, Wenmeng Zhou, and Yingda Chen. SWIFT: A
scalable lightweight infrastructure for fine-tuning. In AAAI-
25, Sponsored by the Association for the Advancement of Ar-
tificial Intelligence, February 25 - March 4, 2025, Philadel-
phia, PA, USA, pages 29733–29735. AAAI Press, 2025. 6
[60] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao,
Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deep-
eyes: Incentivizing ”thinking with images” via reinforce-
ment learning. CoRR, abs/2505.14362, 2025. 2
[61] Mingyang Zhou, Long Chen Yi R. Fung, Christopher
Thomas, Heng Ji, and Shih-Fu Chang. Enhanced chart un-
derstanding in vision and language task via cross-modal pre-
training on plot table pairs. In Findings of ACL, 2023. 3


## 6. Visualization Cases of FinePRM

In this section, we provide qualitative visualizations to as-
sess the scoring capability of our FinePRM. As shown in
Figure 5, 6 and 7, we intuitively evaluate FinePRM through
visualization on complex chart and image scenarios.
Specifically, to produce the scoring heatmaps shown in
the following figures, we define a specific intent and ap-
ply the action across a 32 × 32 image grid. FinePRM then
scores each action independently. The resulting heatmaps
demonstrate that our FinePRM can clearly distinguish be-
tween correct and incorrect regions, assigning the highest
scores to the areas that most accurately fulfill the intent.

## 7. Prompts for Data Generation

In this section, we provide the detailed prompts used in our
pipeline. To ensure the reproducibility of our method, we
present the translated English versions of the prompts used
for Data Enrichment, Intent-Action Synthesis (FinePRM),
and Cold-Start Data Generation (SketchVL Cold Start).

```
Prompts for Data Enrichment and Cleaning
```
1. Chart Attribute Enrichment Prompt
You are a top-tier chart data annotation expert. Your task is to supplement
detailed structured information for each annotation based on the given
chart image and OCR-recognized text labels.
Input Format: Normalized bounding boxes (0-1000).
Task: For each JSON object in the ”Pending Annotations”:
- legendlabel: If the text relates to a legend, fill in the full legend
    text; otherwise, return ””.
- color: Describe the color of the text or associated element (e.g.,
    ”Red”, ”Dark Blue”).
- describe: Detailed description of the role and context (e.g., ”X-axis
    label”, ”Value at the top of a bar”, ”Pie sector label”).
Output: A strictly valid JSON array.
2. General Image Object Verification (Validity)
Please refer to the two images provided (original and cropped crop with
red box). Determine whether the content within the red box constitutes a
meaningful, relatively complete object or a recognizable part of an object
(e.g., a wheel is meaningful; a patch of solid color or sky is not). Answer
only ”Yes” or ”No”.
3. General Image Fine-grained Labeling
Your task is to generate a structured description for the object inside the
red box to uniquely identify it in the original image. Return JSON format:
- label: Precise name of the object (e.g., ”White porcelain plate”).
- relation: Detailed description of the spatial relationship with sur-
    rounding objects to eliminate ambiguity (e.g., ”In the middle of the
    table, below the laptop”).
4. Duplicate Resolution Prompt
Task: Analyze two highly overlapping annotations to determine if they
point to the same object and decide which description is better.
Steps: Check the cropped images and the original image. If they are the
same entity, compare ”Label” and ”Relation” to select the one that is more
accurate and informative. Output: Only the Label name of the retained
annotation.

```
Prompts for Intent-Action Pair Synthesis
```
```
System Prompt:
Your task is to generate training data for a Reward Model. Based on the
given image/chart configuration and annotation information, generate a
JSON list containing N independent ”Action Pairs”.
```
```
Each item must contain:
```
- explanation: A natural language description of the drawing intent.
    It should explain what to mark and where it is, without revealing spe-
    cific coordinates or the fact that you are looking at metadata. (e.g.,
    ”Circle the window with grilles located in the center right of the im-
    age”).
- action: A single drawing instruction JSON object (‘point‘, ‘line‘,
    ‘circle‘, ‘rectangle‘, or ‘text‘) derived accurately from the provided an-
    notations.
Constraints:
- Diversity: The actions must strictly follow a specified sequence (e.g.,
    point - line - circle) and cover different objects.
- Accuracy: Do not guess coordinates. Use the provided annotation data
    strictly.
- Chart Specifics: For charts, X-axis labels should be marked at the top
    edge center; Y-axis labels at the right edge center. Do not mark legends
    if they are not in the annotation.
- Image Specifics: For general images, ensure the description uniquely
    locates the object using spatial relationships provided in the context.
Output Format: A strict JSON list of objects containing ‘explanation‘
and ‘action‘ keys.

```
Prompts for Cold-Start Data Generation
```
1. Visual Question Generation
Please generate a visual question regarding the chart/image content based
on the provided annotations. The question should:
- Focus on the visual content (e.g., values, trends, object relationships,
    counting).
- Be concise and answerable based strictly on the image.
- (For Charts) Focus on topics like: Extremes & Ranking, Comparisons,
    Trends, Statistics.
- (For Images) Focus on topics like: Spatial relationships, Attributes, Ac-
    tions, Context.
Output: Only the question string.
2. Sketch-CoT Reasoning Generation
Your task is to answer the user’s question by generating a multi-step rea-
soning process that includes visual sketching actions.
Output Format: A JSON list of lists, where each sub-list represents a
reasoning turn.
- explanation: Summarize the previous step or plan the current
    drawing. Do not state the final answer until the last step.
- action: A dictionary defining a drawing operation (‘point‘, ‘line‘,
    ‘circle‘, ‘rectangle‘, ‘text‘) with normalized coordinates (0-1000).
Key Instructions:
- Draw then Conclude: You should first perform a drawing action (vi-
    sual grounding) and then discuss the finding in the next turn.
- No Hallucination: Use strictly existing coordinates from the provided
    metadata.
- Chain of Thought: The reasoning should be linear. The first turn plans
    the path; intermediate turns verify data points via drawing; the final turn
    (with empty action) provides the direct answer to the question.
- Conciseness: Solve the problem in typically 3-4 turns.
Input Provided: Chart HTML/Image Annotations and the User Query.


```
Mark the Frequently Angry. Mark the Inspired in the NET category.
```
```
Highlight the Lamb label in the vertical axis. Highlight the Rice label in the vertical axis.
```
```
Highlight the Germany value label in the year 2017. Highlight the US value label in the year 2012.
```
Figure 5. Visualization of FinePRM scoring heatmaps (Case 1). We visualize the spatial credit assignment by FinePRM. The heatmap
represents the score distribution across the image for a given intent. High-scoring regions indicate that FinePRM correctly identifies the
areas most relevant to the instruction.


Highlight the 75 number on the x axis in the image. Mark the mormon on the y axis in the image.

```
Mark the position of cambrain 34B. Mark the legend label of IXC25 in Legend area.
```
```
Circle the heating pipes in the image. Highlight the gray cat in the image.
```
```
Figure 6. Visualization of FinePRM scoring heatmaps (Case 2).
```

```
Mark the Data source in the image. Mark the Don't know segment in the image.
```
```
Mark the character further from the camera. Mark the crab in the image.
```
```
Excellent Acceptable Poor Unacceptable
```
Figure 7. Visualization of FinePRM scoring heatmaps (Case 3). Further visualization of FinePRM’s scoring process on complex charts
and general images. The legend indicates the score quality, where darker colors (e.g., Purple for ‘Excellent’) represent higher scores, and
lighter colors (e.g., Yellow for ‘Unacceptable’) represent lower scores. As shown, the high-scoring regions (darker patches) align precisely
with the semantic intent described below each image, confirming that FinePRM provides accurate, fine-grained feedback.


## 8. Implementation of Fine-Grained Credit Assignment

```
In this section, we provide the implementation logic for our proposed Fine-Grained Policy Optimization (FinePO). Due to
the complexity of the logic, we present the code in a single-column format.
```
### 8.1. Variable Definitions

```
To facilitate the understanding of the provided Python implementation, we map the key variables in the code to the mathe-
matical notations defined in the FinePO methodology.
```
- inputs: The batch data structure containing the generated trajectories{y 1 ,...,yk}, their corresponding FinePRM pro-
    cess scores pj, and associated metadata.
- prioractiondistribution: Represents the fixed prior distribution Q(a) derived from the training set, serving as
    the anchor for the KL divergence constraint.
- actionhistory: A sliding window buffer used to estimate the current policy’s action distribution Pk(a), ensuring a
    stable calculation for the dynamic KL penalty.
- kllambda: Corresponds to the coefficient λKL. It controls the strength of the penalty applied when the current action
distribution deviates from the prior.
- rewardoffsets: Stores the computed clipped KL offset values Oclipped(aj). These offsets are added to the raw
    FinePRM scores to regularize the action types.
- scalaradv: Represents the coarse, cross-trajectory advantage A(yi). This scalar serves as the base value and the total
    ”budget” to be redistributed among the individual steps.
- savg: The length-weighted average score ̄ of all steps within a single response. It acts as the intra-response baseline top
    determine the relative quality ∆jof each step.
- k: The dynamic scaling factor k. It scales the fine-grained adjustment based on the magnitude of the coarse advantage
|A(yi)| and the maximum score deviation.
- alpha: Corresponds to the hyperparameter α. It governs the intensity of the redistribution, determining how much the
fine-grained scores influence the final token-level advantage.
- tokenspans: A utility mapping used to determine the token length Ljand specific indices for each reasoning step sj,
enabling the precise application of the final step-level advantage A(sj) to the token sequence.

```
Algorithm 1: Fine-Grained Advantage Redistribution
```
```
def _prepare_batch_inputs(self, inputs: DataType) -> List[DataType]:
"""
Prepare batch inputs with KL penalty and Token-Level Credit Assignment.
This function runs on each worker process.
"""
# --- Part 1: Global Distribution Alignment (KL Penalty) ---
credit_key = f’credit_details_{self.reward_func_names[0]}’
# 1. Gather local actions from the current batch
local_actions = []
for data in inputs:
if credit_key in data:
# Extract action types (e.g., ’point’, ’line’) from step strings
for step_str in data[credit_key].keys():
action = self._parse_action_type(step_str)
if action in self.prior_action_distribution:
local_actions.append(action)
# 2. Synchronize action counts across all distributed workers
# gathered_actions_flat_list contains actions from ALL GPUs
global_actions = gather_object(local_actions)
reward_offsets = {}
# 3. Calculate KL-based Reward Offsets (Executed on Main Process)
if self.accelerator.is_main_process:
# Update sliding window history with current global counts
global_counts = collections.Counter(global_actions)
self.action_history.append(global_counts)
# Aggregate counts from the sliding window
total_counts = collections.defaultdict(int)
for counts in self.action_history:
for act, c in counts.items():
```

```
total_counts[act] += c
total_n = sum(total_counts.values())
if total_n > 0:
# Calculate current distribution P_curr
p_current = {a: c / total_n for a, c in total_counts.items()}
# Calculate Offset: -lambda*log(P_curr / P_prior)
for action, p_prior in self.prior_action_distribution.items():
p_curr = p_current.get(action, 0.0) + self.epsilon
# KL Divergence direction: Penalty if P_curr > P_prior
offset = -self.kl_lambda*math.log(p_curr / (p_prior + self.epsilon))
# Clip offset to prevent training instability
reward_offsets[action] = clip(offset, -self.clip_val, self.clip_val)
```
# 4. Broadcast calculated offsets to all workers
broadcast_object_list([reward_offsets], from_process=0)

# 5. Apply offsets to local data (Update Reward Scores)
for data in inputs:
if credit_key in data:
for step_str, score in data[credit_key].items():
action = self._parse_action_type(step_str)
# Add the global penalty/bonus to the step score
data[credit_key][step_str] = score + reward_offsets.get(action, 0.0)

# --- Part 2: Token-Level Credit Assignment ---
# Split data into mini-batches for memory efficiency
batches = self.split_by_mini_batches(inputs)
processed_batches = []

for batch in batches:
# (Standard encoding and masking logic omitted for brevity...)
# ...
# Initialize a tensor to hold per-token advantages
# Shape: [Batch_Size, Sequence_Length]
advantages_tensor = torch.zeros_like(labels, dtype=torch.float32)
for j, sample in enumerate(batch):
scalar_adv = sample[’advantages’].item() # Original GAE advantage
credit_details = sample.get(credit_key, {})
# Identify valid completion tokens
mask = (labels[j] != -100)
# Default: Assign the scalar advantage to ALL tokens in the response
per_token_adv = torch.full((mask.sum(),), scalar_adv)
# If FinePRM scores exist, perform redistribution
if credit_details:
# 1. Map logical steps to physical token indices
# Returns: {’step_1’: (start_idx, end_idx), ...}
token_spans = self._map_sub_steps_to_tokens(sample, credit_details)
# 2. Calculate Length-Weighted Average Score (S_avg)
w_score_sum = 0
total_len = 0
for step_str, score in credit_details.items():
if step_str in token_spans:
span_len = token_spans[step_str].len
w_score_sum += score*span_len
total_len += span_len
s_avg = w_score_sum / total_len if total_len > 0 else 0
# 3. Determine Dynamic Scaling Factor (k)
# We calculate deviations from the average
offsets = [score - s_avg for score in credit_details.values()]
max_deviation = max(offsets) if offsets else 0
if max_deviation > 0:
# k scales the deviation so that the max deviation roughly
# corresponds to the magnitude of the scalar advantage.
k = abs(scalar_adv) / (max_deviation + 1e-6)
else:
k = 0


```
# 4. Redistribute Advantage
for step_str, score in credit_details.items():
if step_str not in token_spans: continue
start, end = token_spans[step_str]
offset = score - s_avg
# Core Redistribution Formula:
# A_new = A_original + alpha*k*(Score_step - Score_avg)
adjustment = self.alpha*k*offset
mod_adv = scalar_adv + adjustment
# 5. Conservative Clipping (Advantage Conservation)
# Ensure the sign doesn’t flip excessively and stays within bounds
if scalar_adv < 0: # Negative feedback case
# Allow worse steps to be more negative, but limit improvement
lower_bound = 2.0*scalar_adv
mod_adv = max(lower_bound, min(mod_adv, 0.0))
else: # Positive feedback case
# Allow better steps to be more positive, but limit punishment
upper_bound = 2.0*scalar_adv
mod_adv = max(0.0, min(mod_adv, upper_bound))
# Assign modified advantage to the specific token span
per_token_adv[start:end] = mod_adv
# Fill the tensor
advantages_tensor[j, mask] = per_token_adv
# Update the batch data structure
encoded_batch[’advantages’] = advantages_tensor
processed_batches.append(encoded_batch)
```
return processed_batches


## 9. Evaluation Prompts

To ensure a fair and rigorous comparison, we employ
a model-based judge (DeepSeek-R1-Distill-Qwen-14B) to
evaluate the correctness of SketchVL’s responses. Depend-
ing on the dataset, we apply specific evaluation criteria to
handle the nuances of chart reasoning. Below are the trans-
lated system prompts used for each benchmark.

```
Prompt for ChartQA-Pro Evaluation
```
```
System Instruction:
Please judge the correctness of the following answer. Briefly analyze or
calculate first, and finally append a [true] tag if correct, or [false]
if incorrect. Focus solely on whether the final conclusion matches the
ground truth label. Do not judge the reasoning process.
Evaluation Principles (Relaxed Correctness Metric):
```
1. MCQ & Fact Checking: Use Exact Match. The predicted option
    (A/B/C/D) or Boolean (True/False) must strictly match the label.
2. Year Values: Use Exact Match. Years (e.g., 2010 vs 2009) must be
    precise; visual approximation errors are not allowed for years.
3. Other Numeric Values: Allow a 5% relative error. Calculate error
    E =|P red− GT|/|GT|. If E ≤ 0. 05 , mark as correct. Handle
    unit conversions if necessary.
4. Textual Answers: Assess Semantic Similarity. Exact match is not
    required; ignore minor differences like pluralization (’Female’ vs ’Fe-
    males’) or casing.
5. List Answers: Split the list and evaluate each element individually
    using rules 1-4. All elements must be correct.

```
Prompt for PlotQA Evaluation
```
```
System Instruction:
Analyze the correctness of the response. Output [true] or [false] at
the end. Do not provide an ambiguous ”neither” option.
Evaluation Principles:
```
- Non-Numeric Answers: Analyze semantic understanding. Colors are
    correct if they are semantically similar.
- Numeric Answers: You must perform a calculation to verify the error.
    Compute accuracy acc =|Label− Answer|/|Label|.
- Threshold: If acc≤ 0. 05 , evaluate as True; otherwise False.
- Units: If units do not align, convert them based on context before cal-
    culating the error.
- Formulas: If the assistant outputs an uncomputed formula, evaluate its
    final result against the label with the 5% tolerance.

```
Prompt for EvoChart Evaluation
```
```
System Instruction:
Judge the correctness based on the provided content. Output [true] or
[false].
Evaluation Principles:
```
- Condition **isclear** : If isclear=true is specified in metadata,
    require strictly precise numerical correspondence.
- Condition **not isclear** : If isclear=false, allow a 5% tol-
    erance for all numeric labels.
- Calculation: For numeric labels (e.g., 27 vs 28), if the relative error is
    within±5%, count it as correct. Example: (22− 21)/ 21 < 0. 05 is
    acceptable.
- API Errors: If an API error occurs in the model output, look for a
    correct answer generated prior to the error.

```
Prompt for General, Math and Other Evaluation
```
```
System Instruction:
Judge the correctness of this subjective or general question. Output
[true] or [false].
Evaluation Principles:
```
- Numeric Answers: Allow a 5% tolerance excluding units.
- Years: Be lenient. Treat years as numeric values with 5% tolerance
    (e.g., 2018 vs 2022 is within 5% relative error).
- Non-Numeric: Colors and text need only be semantically close.
- Objective: Focus on the final answer, ignoring intermediate reasoning
    steps unless they contradict the correct final result.

## 10. Training Dynamics Analysis

```
To further validate the effectiveness of our method, we visu-
alize the training dynamics of the SketchVL-3B model and
its ablation variants. Figure 8 presents the curves for Re-
ward, Loss, Learning Rate, Gradient Norm, KL Divergence,
and Mean Completion Length over the training steps.
```
```
Stability Analysis. As observed in the Gradient Norm
and KL Divergence curves, the full SketchVL-3B model
(Figure 8a) exhibits significantly higher training stability
compared to the baselines.
```
- The w/o FinePO (naive GRPO) model (Figure 8b),
    which relies on coarse trajectory-level rewards (Naive
    GRPO), shows drastic spikes in Gradient Norm and un-
    stable KL divergence. This confirms that uniformly
    broadcasting rewards to all tokens introduces substantial
    noise, destabilizing the policy update.
- Similarly, the w/o Sketch (zero GRPO) model (Fig-
    ure 8c) lacks the explicit grounding of reasoning steps,
    leading to a harder optimization landscape. Notably, al-
    though it achieves a relatively ideal training reward, it per-
    forms poorly on test benchmarks.
In contrast, our FinePO mechanism (Figure 8a) effectively
smooths the learning process by assigning precise credit
to individual steps, resulting in stable gradient updates and
controlled policy deviation.

```
Reward Analysis. It is worth noting that the final con-
verged Reward value of the full SketchVL model is slightly
lower than that of the baselines (e.g., w/o FinePO). This
is an expected and reasonable outcome. Our FinePO algo-
rithm introduces strict constraints—specifically the KL Ac-
tion Regularization and the FinePRM alignment—to pre-
vent ”reward hacking” (where the model exploits loopholes
to get high scores without proper reasoning). While these
additional rules and constraints slightly reduce the absolute
scalar value of the reward, they force the model to adhere
to a more rigorous and generalizable reasoning logic, ulti-
mately leading to better performance on actual benchmarks
as shown in our main experiments.
```

```
(a) Full SketchVL-3B (Ours). Trained with FinePO and FinePRM. (b) w/o FinePO (Naive GRPO). Uses only coarse trajectory-level rewards.
```
```
(c) w/o Sketch (Zero GRPO). No interactive reasoning actions. (d) w/o FinePRM (Random Scores). Credit is assigned randomly.
```
Figure 8. Comparison of Training Dynamics. We visualize the training metrics for the full SketchVL-3B model and three ablation
baselines. The curves include Reward (mean training reward), Loss, Learning Rate, Gradient Norm, KL Divergence (between the current
policy and the reference model), and Mean Completion Length.


