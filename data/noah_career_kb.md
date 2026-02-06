# Noah's Career & Background Knowledge Base

This file contains detailed information about Noah's professional experience, technical skills, projects, and background.
Each section is designed to be useful for RAG retrieval and semantic search.

---

## Projects Summary

### Project 1: Portfolia AI Assistant

Portfolia is Noah's flagship project — an AI-powered portfolio assistant built with a RAG (Retrieval-Augmented Generation) architecture. The tech stack includes LangGraph for stateful conversation orchestration, Supabase with pgvector for vector storage and semantic retrieval, FastAPI for the backend API, and OpenAI embeddings for semantic search. It demonstrates production-grade skills in AI/ML engineering, prompt engineering, database design, API development, and error handling. Noah built it from scratch as both a portfolio showcase and a working example of enterprise AI architecture.

### Project 2: Generic Lead Response Heatmap Dashboard

Noah built a Python-based heatmap dashboard that visualizes lead response time patterns. It's designed as a generic, reusable tool — it uses a sample dataset to demonstrate how any sales team can identify coverage gaps and optimize when leads are being contacted. The dashboard uses Python with pandas for data processing and matplotlib/seaborn for heatmap visualization. What makes it notable is that Noah saw a real operational problem — teams not knowing when their response coverage was weakest — and built a generalizable solution that works with any team's data.

### Project 3: Employee Attrition Prediction Model (DATA 430)

For his DATA 430 coursework at UNLV, Noah built a logistic regression model predicting employee attrition. He achieved 94.75% accuracy and his professor specifically called out his "real modeling discipline." The project demonstrated skills in logistic regression, Bayesian classification, feature engineering, and statistical analysis. It wasn't just an academic exercise — Noah approached it like a real business problem.

---

## Tesla Career

### Inside Sales Advisor Role

Noah works as an Inside Sales Advisor at Tesla in Las Vegas. He achieved Q3 Plaid Club Top Performer recognition, placing him in the top 10% of the sales team. This recognition demonstrates his ability to perform at a high level in a competitive, metrics-driven environment.

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
- Make decisions under pressure with imperfect data
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

This foundation trained him to reason quantitatively in complex systems and made it easier to later apply statistics and data analysis in non-academic domains like logistics, real estate, and machine-learning coursework.

---

## GitHub Project: Portfolia AI Assistant

### Repository and Overview

Repository: https://github.com/iNoahCodeGuy/Noahs_Assistant.git

Portfolia is Noah's flagship project — a production-grade AI-powered portfolio assistant built with a 22-node LangGraph pipeline. It uses RAG (Retrieval-Augmented Generation) to answer questions about Noah's professional background, technical skills, and projects.

### Technical Architecture

Portfolia's 22-node pipeline (assistant/flows/conversation_flow.py):
- **Orchestration**: LangGraph for stateful conversation management
- **Retrieval**: Supabase pgvector with text-embedding-3-small (1536 dimensions)
- **Generation**: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929) at temperature 0.7
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
- LangGraph pipeline orchestration (22 nodes)
- LLM prompt engineering and response formatting
- Python backend development
- FastAPI API design
- Production error handling and observability (LangSmith)

---

## GitHub Project: Predicting Employee Attrition

### Repository and Overview

Repository: https://github.com/iNoahCodeGuy/Predicting-Employee-Attrition-Using-Logistic-Regression.git

This is a machine learning project that predicts whether employees will leave a company based on various features like satisfaction level, number of projects, average monthly hours, time spent at company, work accidents, promotions, department, and salary.

### Problem Statement

Employee attrition is costly for organizations. Being able to predict which employees are at risk of leaving allows companies to intervene proactively—whether through retention programs, workload adjustments, or compensation changes.

### Technical Approach

Noah used:
- Algorithm: Logistic regression for binary classification
- Data preprocessing: Feature engineering, handling categorical variables, normalization
- Model evaluation: Accuracy, precision, recall, F1-score, ROC-AUC
- Interpretation: Analyzed coefficient weights to understand which factors most influence attrition

### Key Findings

The model identified satisfaction level, number of projects, and average monthly hours as the strongest predictors of employee attrition. Employees with low satisfaction, high project loads, and long hours were significantly more likely to leave.

### Skills Demonstrated

This project demonstrates:
- Understanding of supervised learning and classification problems
- Feature engineering and data preprocessing
- Model evaluation and interpretation (not just accuracy, but understanding what drives predictions)
- Ability to translate statistical results into business insights
- Python (pandas, scikit-learn, NumPy)
- Logistic regression implementation and tuning
- Statistical evaluation metrics
- Data visualization for model interpretation

---

## GitHub Project: Response Time Call Center Analysis

### Repository and Overview

Repository: https://github.com/iNoahCodeGuy/response_time_cl_analysis.git

This is a statistical analysis of call center response time performance, examining whether response times differ significantly across different periods or conditions.

### Business Context

In sales and customer service environments, response time directly impacts conversion rates and customer satisfaction. This project analyzes response time patterns to identify when performance degrades and what might be causing it.

### Technical Approach

Noah used:
- Statistical testing: Hypothesis testing to determine if response time differences are statistically significant
- Data analysis: Descriptive statistics, distribution analysis, outlier detection
- Visualization: Time-series plots, distribution plots, comparative analysis
- Insights: Identified specific time windows or conditions where response times spike

### Problem Solved

Rather than relying on anecdotal evidence ("we feel slower on Mondays"), this provides statistical proof of performance patterns. It enables data-driven decisions about staffing, process improvements, or system optimizations.

### Skills Demonstrated

This project demonstrates:
- Hypothesis-driven analysis
- Statistical significance testing
- Time-series data analysis
- Translating statistical results into operational recommendations
- Understanding of experimental design and controls
- Python (pandas, scipy.stats, matplotlib/seaborn)
- Data cleaning and preparation
- Exploratory data analysis (EDA)

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

Noah builds APIs with FastAPI, designs database schemas, implements production error handling and logging, and optimizes systems (achieving 4.3x performance improvement in Portfolia).

### Development Tools

Noah uses Jupyter notebooks for analysis, VS Code for development, LangSmith for LLM observability, and Supabase for backend infrastructure.

---

## Combat Sports Background

### Current Role and Rank

Noah coaches kids BJJ and the amateur MMA team at Xtreme Couture. He holds a purple belt in Brazilian Jiu-Jitsu.

### Professional Impact of Combat Sports

Combat sports have had a direct impact on how Noah operates professionally.

Discipline under discomfort: Progress only happens through consistent, unglamorous work—showing up tired, drilling fundamentals, and trusting process over emotion. That carries over into work where results compound quietly long before they're visible.

Problem-solving under pressure: In a fight, plans rarely survive first contact. You're forced to adapt in real time, recognize patterns, and make decisions with incomplete information. That's the same mindset Noah uses in professional environments where conditions change quickly and perfect data doesn't exist.

Emotional regulation: If you panic, rush, or react impulsively, you get punished immediately. Learning to stay calm, assess, and respond deliberately has translated into clearer communication and better decision-making in high-stress situations.

Accountability: There's no one else to blame in a fight. Preparation, execution, and outcomes are yours. That mindset carries into Noah's professional life—he takes ownership of results, especially when things go wrong, and focuses on improving the process rather than externalizing responsibility.

Overall, combat sports trained Noah to stay composed, adaptable, and accountable in environments where pressure, uncertainty, and consequences are real.

---

## Career Goals

### Target Roles

Noah is targeting roles as a Data Analyst, Business Intelligence Analyst, or Software Engineer. His ideal role in 2-3 years is a technical data analyst or software engineering position.

### What Differentiates Noah

Noah is not coming from theory alone—he's coming from operating in real, high-stakes systems where decisions have consequences and uncertainty is constant.

He has spent years making decisions with incomplete information in logistics, real estate, and sales. That trained him to think in terms of risk, trade-offs, and incentives, not just idealized solutions. When he learns technical concepts, he naturally maps them to real systems and failure modes rather than treating them as abstract exercises.

Noah has a strong foundation in quantitative reasoning from his biology background. He was trained to think in hypotheses, evaluate evidence, and avoid over-interpreting noisy data. That makes it easier for him to understand statistics, machine learning, and model limitations—not just how to use tools, but when they break.

Unlike many career switchers, Noah is already comfortable with production pressure. He has owned outcomes, handled failures publicly, and been accountable to customers and stakeholders in real time. That translates directly to engineering environments where systems fail, priorities shift, and clear communication matters as much as clean code.

Noah has proven he can learn hard things in demanding environments. He maintained high performance in his sales role (Top 10% at Tesla) while separately building technical projects and completing graduate analytics coursework. That signals adaptability, endurance, and the ability to grow without needing perfect conditions.

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

## Contact Information

- **Location**: Las Vegas, Nevada
- **GitHub**: https://github.com/noahxl10
- **LinkedIn**: https://linkedin.com/in/noahcuellar
- **Email**: [Contact via Portfolia chat interface]
- **Location**: Las Vegas, NV

Noah is open to opportunities in data analysis, business intelligence, and software engineering roles where he can apply his unique combination of technical skills, business operations experience, and quantitative reasoning.
