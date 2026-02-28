# Noah's Career & Background Knowledge Base

This file contains detailed information about Noah's professional experience, technical skills, projects, and background.
Each section is designed to be useful for RAG retrieval and semantic search.

---

## Projects Summary

### Project 1: Portfolia AI Assistant

Portfolia is Noah's flagship project — an AI-powered portfolio assistant built with a RAG (Retrieval-Augmented Generation) architecture. The tech stack includes LangGraph for stateful conversation orchestration, Supabase with pgvector for vector storage and semantic retrieval, FastAPI for the backend API, and OpenAI embeddings for semantic search. It demonstrates production-grade skills in AI/ML engineering, prompt engineering, database design, API development, and error handling. Noah built it from scratch as both a portfolio showcase and a working example of enterprise AI architecture.

### Project 2: Generic Lead Response Heatmap Dashboard

Noah built a Python-based heatmap dashboard that visualizes lead response time patterns. It's designed as a generic, reusable tool — it uses a sample dataset to demonstrate how any sales team can identify coverage gaps and optimize when leads are being contacted. The dashboard uses Python with pandas for data processing and matplotlib/seaborn for heatmap visualization. What makes it notable is that Noah saw a real operational problem — teams not knowing when their response coverage was weakest — and built a generalizable solution that works with any team's data.

### Project 3: Employee Attrition Prediction Model

Noah built a logistic regression model predicting employee attrition. He achieved 94.75% accuracy through rigorous methodology. The project demonstrated skills in logistic regression, Bayesian classification, feature engineering, and statistical analysis. Noah approached it like a real business problem — identifying retention risk factors and building a model that could actually inform HR decisions.

---

## Tesla Career

### Inside Sales Advisor Role

Noah works as an Inside Sales Advisor at Tesla in Las Vegas. He achieved Plaid Club recognition, placing him in the top 10% of the sales team. This recognition demonstrates his ability to perform at a high level in a competitive, metrics-driven environment.

### Why Transitioning to Tech

Noah's transition from sales to tech is driven by a desire to work upstream on the systems that create leverage. As he explains: "Sales taught me how the business actually works—how customers think, where friction shows up, and what decisions really matter. Over time, I realized I was more interested in building the systems behind the decisions than just executing the pitch."

He kept noticing inefficiencies—manual workflows, slow feedback loops, data that existed but wasn't being used. Rather than accepting these as constraints, he started teaching himself how to work with data and software to fix those problems.

"I'm not leaving sales because I'm bad at it. I'm moving into tech because I want to operate upstream, where better tools and models create leverage for thousands of decisions instead of one conversation at a time."

---

## Total Quality Logistics Experience

### Logistics Account Executive

At Total Quality Logistics, Noah worked as a Logistics Account Executive, managing relationships between companies that needed to move freight and the trucking market that actually moved it. He acted as the single point of accountability for his customers' shipments.

### Day-to-Day Responsibilities

Noah's daily work involved:
- Sourcing and developing shipper accounts, learning their business, and handling their transportation needs end-to-end
- Understanding their lanes, volumes, seasonality, and service requirements
- Pricing freight in real time based on market conditions
- Securing reliable carriers to haul the loads
- Tracking shipments and solving problems when disruptions happened

### The Challenge of Real-Time Decision Making

The trucking market is essentially a live supply-and-demand system. When demand is high and trucks are scarce, rates spike. When trucks are plentiful and demand is low, rates drop. Noah had to price in real-time factoring in: how many trucks are available in that region, fuel costs, driver willingness to run the lane, weather/holidays/port congestion, and how fast the load needed to move.

The hardest part was uncertainty. He often had to make commitments before knowing whether a carrier would cancel last minute, whether a warehouse would delay loading, or whether market conditions would shift overnight. When problems happened, his job was to absorb the stress, communicate clearly, and keep the customer's operation running without escalation.

### Skills Developed in Logistics

From his logistics experience, Noah learned to:
- Think in systems, not single transactions
- Price freight in real-time against fluctuating market conditions
- Balance short-term margins against long-term relationships
- Communicate clearly during high-stress situations

It was a mix of sales, operations, and real-time decision-making—closer to managing a live system than following a script.

---

## Signature Real Estate Group Experience

### Real Estate Agent Role

At Signature Real Estate Group, Noah worked as a Real Estate Agent. He acted as a market interpreter, negotiator, and risk manager for clients going through one of the highest-stakes financial decisions they'll ever make. He represented buyers and sellers throughout the entire transaction lifecycle.

### Core Responsibilities

Noah's work included:
- Educating clients on market conditions before they ever made a decision
- Pricing properties based on real-time supply, demand, and buyer behavior
- Marketing listings and qualifying buyers
- Negotiating contracts, contingencies, and timelines
- Coordinating with lenders, inspectors, appraisers, escrow, and title

### Pricing as Probabilistic Decision-Making

The housing market is driven by supply (how many homes are available), demand (how many qualified buyers are searching), and financing (interest rates, lending standards, buyer affordability). Small changes in any one can dramatically shift outcomes.

Pricing a home isn't about what a seller wants—it's about where buyers will react. If priced too high, it sits and loses momentum. If priced strategically, it creates urgency and can generate multiple offers. Noah had to assess recent comparable sales, days on market trends, buyer sentiment, financing friction, and seasonality—then translate that into a recommendation that balanced maximizing value with actually getting the deal done.

### Managing High-Stakes Transactions

Once under contract, risk increases. Deals fall apart because of appraisal gaps, inspection findings, financing delays, buyer cold feet, and emotional decision-making. Noah's job wasn't just to negotiate terms—it was to manage expectations, emotions, and incentives so rational decisions could still happen under stress.

### Skills Developed in Real Estate

From real estate, Noah learned to:
- Read markets, not just prices
- Negotiate when information is asymmetric
- Translate complex financial concepts into plain language
- Manage high-stress interpersonal dynamics
- Balance logic with psychology

At its core, real estate was about decision-making under uncertainty with permanent consequences.

---

## Education Background

### Bachelor of Science in Biology from UNLV

Noah earned a Bachelor of Science in Biology from the University of Nevada, Las Vegas (UNLV) in 2018. The program emphasized scientific reasoning, data analysis, and evidence-based decision making. While his degree wasn't formally labeled biostatistics, biology as a discipline is inherently quantitative.

### Quantitative and Biostatistics Skills

Through his biology degree, Noah developed:
- Strong understanding of descriptive statistics, including distributions, averages, variance, and outliers
- Hypothesis-driven thinking—framing questions as testable claims rather than assumptions
- Interpreting statistical results conceptually, including p-values, confidence, and experimental uncertainty
- Evaluating signal vs. noise in real-world data where clean results are rare
- Identifying confounders, bias, and the importance of proper experimental controls

This foundation trained him to reason quantitatively in complex systems and made it easier to later apply statistics and data analysis in domains like logistics, real estate, and machine learning.

---

## GitHub Project: Portfolia AI Assistant

### Repository and Overview

Repository: https://github.com/iNoahCodeGuy/Noahs_Assistant.git

Portfolia is Noah's flagship project — a production-grade AI-powered portfolio assistant built with a LangGraph pipeline. It uses RAG (Retrieval-Augmented Generation) to answer questions about Noah's professional background, technical skills, and projects.

### Technical Architecture

Portfolia's pipeline (assistant/flows/conversation_flow.py):
- **Orchestration**: LangGraph for stateful conversation management
- **Retrieval**: Supabase pgvector with text-embedding-3-small (1536 dimensions)
- **Generation**: Anthropic Claude Sonnet 4.5 (claude-sonnet-4-5-20250929), with Claude Haiku for intent classification
- **Backend**: Python with FastAPI for API endpoints
- **State Management**: TypedDict-based ConversationState with 46 fields
- **Observability**: LangSmith tracing for all LLM calls

### Deep Dive: How Retrieval Works

When you ask Portfolia a question, here's what happens:
1. Your query is embedded using OpenAI text-embedding-3-small (1536 dimensions)
2. Supabase RPC function `match_kb_chunks` runs native pgvector similarity search
3. Two thresholds: 0.5 (strict) and 0.3 (fallback for broader recall)
4. Top-k chunks (usually 3-4) are returned with similarity scores
5. Chunks are assembled into context for the LLM

### Deep Dive: The Pipeline Stages

Stage 0: Initialize state (all 46 fields)
Stage 1: Handle greetings + classify intent (knowledge vs crush confession vs off-topic)
Stage 2: Classify role, detect query intent, extract entities
Stage 3: Assess clarification needs, compose optimized query
Stage 4: Retrieve chunks from pgvector, validate grounding sufficiency
Stage 5: Generate draft answer, validate quality, check hallucinations
Stage 6: Plan actions (hiring signals, resume requests), format answer with follow-ups
Stage 7: Execute actions, update memory, log to Supabase

### Key Technical Decisions

- **Intent classification before RAG**: Crush confessions, greetings, and off-topic bypass retrieval entirely (saves compute, better UX)
- **Two-tier similarity thresholds**: 0.5 for precision, 0.3 for recall when needed
- **Bounded memory**: Supports 100+ turn conversations with sliding window (last 4 responses for novelty checks)
- **Graceful degradation**: If retrieval fails, continues with empty context instead of crashing
- **Role-aware prompts**: Different tone and depth for developers vs hiring managers

### Error Handling Patterns

- Graceful degradation: External service failures don't crash the pipeline
- Grounding validation: If similarity scores are too low, admits lack of information
- Quality gates: Non-blocking checks at retrieval and generation stages
- All errors logged to Supabase for observability

### Problem Solved by Portfolia

Traditional portfolios are static. Portfolia allows dynamic, conversational exploration of Noah's background—answering specific questions recruiters or hiring managers might have without requiring them to read through dense PDFs or LinkedIn profiles.

### Technical Skills Demonstrated

The Portfolia project demonstrates Noah's skills in:
- RAG architecture design and implementation
- Vector database optimization (pgvector)
- LangGraph pipeline orchestration
- LLM prompt engineering and response formatting
- Python backend development
- FastAPI API design
- Production error handling and observability (LangSmith)

---

## GitHub Project: Predicting Employee Attrition

### Repository and Overview

Repository: https://github.com/iNoahCodeGuy/Predicting-Employee-Attrition-Using-Logistic-Regression.git

This is a machine learning project that predicts whether employees will leave a company using the Kaggle "Employee Future Prediction" dataset (4,653 rows, 9 columns). Features include Education, JoiningYear, City, PaymentTier, Age, Gender, EverBenched, and ExperienceInCurrentDomain — with LeaveOrNot as the binary target (34.4% left, 65.6% stayed).

### Technical Approach

Noah used Python with pandas, NumPy, scikit-learn, matplotlib, and seaborn. The pipeline includes:
- One-hot encoding of categorical variables (Education, City, Gender, EverBenched) with drop_first=True to avoid the dummy variable trap
- 80/20 train/test split with stratification to preserve class balance
- StandardScaler fit only on training data to prevent data leakage
- Logistic regression with GridSearchCV hyperparameter tuning (C values from 0.001 to 100, L1/L2 penalty, liblinear solver, 5-fold cross-validation)

### Results

- Accuracy: 79.1% on the test set
- Precision: 73.7%
- Recall: 54.0% (acknowledged as a limitation — model misses many actual attritions)
- F1 Score: 0.623
- AUC-ROC: 0.725

### Key Findings

The model revealed a striking gender disparity: 47.1% of female employees left versus 25.8% of males. Location mattered significantly — Pune had 50.4% attrition compared to 26.7% in Bangalore. Payment tier and education level were also strong predictors. Noah converted coefficients to odds ratios for interpretability and visualized predicted probability distributions to understand the classification boundary.

### Skills Demonstrated

This project demonstrates:
- Logistic regression for binary classification with hyperparameter tuning via GridSearchCV
- Proper train/test methodology (stratified split, scaling without leakage)
- Model evaluation beyond accuracy (precision, recall, F1, ROC-AUC, confusion matrix)
- Feature importance analysis through coefficient interpretation and odds ratios
- Data visualization (confusion matrix heatmap, ROC curve, coefficient bar charts, probability distributions)
- Python (pandas, NumPy, scikit-learn, matplotlib, seaborn)

---

## GitHub Project: Response Time CL Analysis

### Repository and Overview

Repository: https://github.com/iNoahCodeGuy/response_time_cl_analysis.git

This project has a personal origin — from Noah's first phone sales job at UFC FIT, he always wondered what the actual impact of response time was on close rates. Using statistical inference, he built a program that can investigate this relationship. His assumption is that results would differ across industries, making it a flexible analytical tool.

### Technical Approach

Python with pandas for data processing, statistical analysis including confidence intervals and hypothesis testing. The sample data in the program is AI-generated, but any dataset can be plugged in — it's designed as a reusable analytical framework.

### Skills Demonstrated

This project demonstrates:
- Statistical inference and hypothesis testing
- Confidence interval analysis
- Reusable analytical framework design
- Python (pandas, scipy.stats, matplotlib/seaborn)
- Data-driven investigation of business questions
- Translating personal observations into testable hypotheses

---

## GitHub Project: Generic Lead Response Heatmap

### Repository and Overview

Repository: https://github.com/iNoahCodeGuy/generic-lead-response-heatmap.git

Noah built a Python-based heatmap dashboard that visualizes lead response time patterns. It's designed as a generic, reusable tool — it uses a sample dataset to demonstrate how any sales team can identify coverage gaps and optimize when leads are being contacted.

### Technical Implementation

The data pipeline processes raw lead data (from a sample dataset) into a time-based grid showing response patterns by hour and day of week. Python with pandas handles data processing, and matplotlib/seaborn generate the heatmap visualization.

### What Makes It Impressive

Noah saw a real operational problem — teams not knowing when their response coverage was weakest — and built a generalizable solution. The tool is designed to work with any team's data, not just one specific use case. It demonstrates initiative: seeing a gap and building a solution rather than waiting to be asked.

### Technical Details

- Data loading and transformation with pandas
- Time-based aggregation (hour of day, day of week)
- Heatmap visualization with matplotlib/seaborn
- Sample dataset included for demonstration
- Clean separation between data processing and visualization

### Skills Demonstrated

This project demonstrates:
- Building practical business intelligence tools
- Data aggregation and transformation for visualization
- Python (pandas, matplotlib/seaborn)
- Heatmap visualization techniques
- Git version control

### Deep Dive: How the Data Pipeline Works

When you ask Portfolia about how this tool works in detail, here's what happens under the hood:

1. **Data Loading**: The sample dataset is loaded via pandas read_csv(). The data contains timestamps, response times, and categorical fields.

2. **Time Extraction**: Response timestamps are parsed and decomposed into hour-of-day (0-23) and day-of-week (0-6, Monday-Sunday).

3. **Aggregation**: Using pandas groupby(), response times are aggregated by the hour/day grid. The aggregation function (mean, median, or count) can be configured.

4. **Pivot Table Creation**: The aggregated data is reshaped into a 24x7 matrix (hours as rows, days as columns) using pandas pivot_table().

5. **Heatmap Rendering**: matplotlib/seaborn's heatmap() function visualizes the pivot table with a color gradient. Darker colors indicate slower response times or gaps in coverage.

This is a generic pattern that works with any time-series response data — swap in your own dataset and the same pipeline produces insights for your team.

---

## Technical Skills Summary — Noah's Technical Background

This section covers Noah's technical background, technical skills, programming languages, coding abilities, and what he can code. Noah's technical background spans programming, data analysis, machine learning, and AI systems. His technical skills include full-stack development, backend APIs, database design, and production system optimization.

### Programming Languages and Technical Skills

**Primary programming language**: Python is Noah's main weapon — he's built analytics dashboards, data pipelines, and the RAG architecture powering Portfolia. He has intermediate to advanced Python skills with libraries including pandas, NumPy, scikit-learn, matplotlib, and seaborn.

**SQL and databases**: Noah writes SQL for query writing, data extraction, joins, and aggregations. He's experienced with PostgreSQL and Supabase.

**Version control**: He uses Git and GitHub for version control, repository management, and collaborative workflows.

### Machine Learning and Statistics

Noah has experience with logistic regression and classification problems, hypothesis testing and statistical significance, model evaluation using accuracy, precision, recall, F1, and ROC-AUC metrics, and feature engineering and data preprocessing.

### Data Visualization and Business Intelligence

Noah builds heatmaps, time-series plots, and distribution analyses. He develops dashboards for business stakeholders and translates data into actionable insights.

### AI and LLM Systems

Noah designs and implements RAG architecture, works with vector databases (Supabase, pgvector), integrates LLMs (OpenAI API), and performs prompt engineering and response optimization.

### Backend Development

Noah builds APIs with FastAPI, designs database schemas, implements production error handling and logging, and optimizes systems for production deployment.

### Development Tools

Noah uses Jupyter notebooks for analysis, VS Code for development, LangSmith for LLM observability, and Supabase for backend infrastructure.

---

## Combat Sports Background

### Coaching at Xtreme Couture

Noah's coaching journey started when he was an assistant coach for his younger brother's high school wrestling team. When the head kids coach at Xtreme Couture reached out to him, it felt like a natural progression — he now coaches kids ages 6-12 in BJJ. Additionally, when Noah retired from competing in MMA, he started coaching amateur fighters and corners them on the regional circuit.

His coaching philosophy centers on teaching the higher purpose behind each action rather than drilling memorized sequences. His reasoning: strict memorization has low retention, and any variation from the sequence during live competition can throw off an athlete. He believes in understanding WHY a technique works so athletes can adapt in real time. He sees this as an allegory for other aspects of life — understanding principles over memorizing steps.

### MMA Fighting Career

Noah is a veteran of 10 MMA fights: 8 amateur (including 2 amateur title fights, one of which he won by defeating 5-0 Edgar Sorto for the Fierce Fighting Championship's 135lb title) and 2 professional fights. He holds a purple belt in Brazilian Jiu-Jitsu.

---

## What Differentiates Noah

Noah is not coming from theory alone—he's coming from operating in real, high-stakes systems where decisions have consequences and uncertainty is constant.

He spent 16 months managing freight logistics at TQL — coordinating carriers, negotiating rates, and solving routing problems under tight deadlines. In real estate, he handled end-to-end transactions where a missed detail could kill a deal. At Tesla, he closed vehicle sales and earned Plaid Club top 10% recognition. Each role required weighing trade-offs quickly and owning the outcome. When he learns technical concepts, he maps them to real systems and failure modes rather than treating them as abstract exercises.

Noah has a strong foundation in quantitative reasoning from his biology background. He was trained to think in hypotheses, evaluate evidence, and avoid over-interpreting noisy data. That makes it easier for him to understand statistics, machine learning, and model limitations—not just how to use tools, but when they break.

Unlike many career switchers, Noah is already comfortable with production pressure. He has owned outcomes, handled failures publicly, and been accountable to customers and stakeholders in real time. That translates directly to engineering environments where systems fail, priorities shift, and clear communication matters as much as clean code.

Noah has proven he can learn hard things in demanding environments. He maintained high performance in his sales role (Top 10% at Tesla) while separately building technical projects. That signals adaptability, endurance, and the ability to grow without needing perfect conditions.

Noah is not transitioning into tech to escape pressure. He is moving into it because his skill set is already aligned with how real technical systems are built, evaluated, and improved.

---

## Learning Philosophy

### Structured and Iterative Approach

Noah's approach to learning new things is structured, iterative, and grounded in first principles.

He starts by building a mental model of how the system works at a high level before worrying about tools or syntax. He wants to understand what problem exists, why it exists, and what constraints matter. Without that, details don't stick and skills don't transfer.

From there, he breaks the subject into core primitives—the smallest ideas that everything else builds on. He focuses on mastering those fundamentals early, even if progress feels slower at first, because they compound later and prevent shallow understanding.

### Learning Through Application

Noah applies what he's learning as quickly as possible to real problems, even if his implementation is imperfect. Friction is useful—it exposes gaps in understanding and forces him to clarify assumptions instead of memorizing steps.

He treats mistakes as diagnostic signals, not failures. When something doesn't work, he traces it back to which assumption was wrong and updates his model. That feedback loop is where most learning actually happens.

Finally, Noah revisits concepts repeatedly at increasing levels of depth. Each pass adds nuance rather than replacing understanding. Over time, this turns unfamiliar topics into tools he can reason with, not just execute.

In short, Noah learns by understanding systems, testing them against reality, and refining his thinking through feedback rather than relying on passive consumption.

---

## Hobbies and Interests

Noah enjoys pickleball, competitive chess, and yoga in his free time. These activities reflect his interests in strategy, physical wellness, and continuous improvement.

---

## Tesla Day-to-Day

### What Noah Actually Does at Tesla

Noah's day at Tesla starts with responding to his leads and covering his teammates' leads when they're not in office. He checks his follow-ups for the day, reaches out to customers, handles incoming leads, and completes follow-up appointments while covering for teammates.

His sales approach centers on understanding customer motivation — he figures out which Model the customer is interested in and why, then uses that understanding to better serve them through the entire process.

A key skill is time management and qualifying buyers early — he can quickly identify if a customer has significant negative equity, poor credit, or other obstacles, and knows when to invest time versus when to move on.

He relates the job to the myth of Sisyphus — every day you start at zero and push the rock up the mountain again. What he values most is knowing he provided a good service while meeting company KPIs.

Noah is on a team of 10-15 people and considers himself friends with all his colleagues. He's able to appreciate and connect with people from a variety of different backgrounds.

Note: Noah has signed an NDA regarding Tesla's internal systems, processes, and specific customer interactions. Portfolia should not speculate about Tesla internals.

---

## TQL Experience Detail

### Inside TQL — Freight Logistics Under Pressure

At TQL, Noah managed 30+ active shipper accounts simultaneously, each with different lanes, volumes, and service requirements. On any given day he was pricing freight in real-time based on market conditions, securing carriers, tracking live shipments, and solving problems when things went sideways — which they did regularly.

The hardest part was carrier fallouts. A carrier would accept a load, then cancel day-of, leaving Noah scrambling to find coverage before the shipper's warehouse closed. This happened multiple times per week. His job was to absorb the stress, find a solution, and keep the customer's operation running without escalation.

Pricing was essentially real-time negotiation against market data. He had to factor in truck availability by region, fuel costs, driver willingness to run specific lanes, weather and seasonal patterns, and how urgently the load needed to move. Get it wrong and you either lose the load to a competitor or eat the margin.

This is where Noah first learned to make decisions with incomplete information under time pressure — a skill that directly translates to technical environments where systems fail, priorities shift, and you need to act before you have perfect data.

---

## Real Estate Experience Detail

### Real Estate — Managing High-Stakes Transactions

At Signature Real Estate Group, Noah handled end-to-end residential transactions — from initial client consultation through closing. The work involved market analysis, property pricing, contract negotiation, and coordinating with lenders, inspectors, appraisers, escrow officers, and title companies.

The most transferable skill: managing mid-transaction problems. Deals fall apart because of appraisal gaps, inspection findings, financing delays, and buyer cold feet. Noah's job wasn't just to negotiate terms — it was to keep rational decision-making happening when emotions were running high and stakes were real (people's homes, life savings, timelines).

He learned to coordinate across 5-7 different parties who all had different incentives, timelines, and communication preferences. That's essentially stakeholder management — keeping everyone aligned on shared outcomes when each party optimizes for different things.

---

## Learning Path and Origin Story

### How Noah Learned to Code

Noah started coding in August 2024 — less than two years ago. He didn't come from a CS background or a bootcamp. He started because he noticed his daily screen time was 8+ hours and decided if he was going to stare at a screen that long, it should be building something.

The catalyst was actually chess. Noah is a competitive chess player, and the story of AlphaZero beating Stockfish fascinated him — a system that learned purely through self-play, without any human chess knowledge, and developed completely novel strategies that grandmasters had never seen. That intersection of strategy, learning systems, and AI is what pulled him in.

He started with Python fundamentals, then moved into data analysis (pandas, NumPy), then statistical modeling (scikit-learn), then into AI/ML systems. Within months he was building Portfolia — a full RAG pipeline — as a way to learn by shipping, not just studying.

### Certifications and Formal Training

- IBM AI Engineering Professional Certificate
- IBM AI Developer Professional Certificate
- IBM Data Science Professional Certificate
- Applied Machine Learning (94.75% grade)

What's notable isn't the certifications themselves — it's the velocity. Noah went from zero coding experience to building production AI systems in under a year while maintaining top 10% sales performance at Tesla. That signals learning speed, not just effort.

---

## Portfolia Development Challenges

### Hardest Part of Building Portfolia

When asked about the hardest part of building Portfolia, Noah says it's dealing with edge cases gracefully and minimizing hallucinations. That's the real engineering challenge — not getting the happy path working, but handling all the ways a conversation can go sideways while still giving accurate, grounded responses.

### What's Next for Portfolia

Noah's next goal for Portfolia is giving it access to the internet. If someone asks a question outside the knowledge base, instead of just saying "I don't know," Portfolia would be able to say something like "That's not in my knowledge base — want me to look it up for you?" This would make Portfolia a more complete assistant rather than being limited to just what's been embedded.

---

## Contact Information

- **Location**: Las Vegas, Nevada
- **GitHub**: https://github.com/iNoahCodeGuy
- **LinkedIn**: https://www.linkedin.com/in/noah-de-la-calzada-250412358/
- **Email**: [Contact via Portfolia chat interface]
- **Location**: Las Vegas, NV

