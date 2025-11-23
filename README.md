📌 Project Overview

This project implements a simplified Airline Crew Scheduling System using backtracking (constraint satisfaction) in Python.
The objective is to assign flights to crew members such that:

Every flight has at least one crew member

No crew member handles overlapping flights

Crew members get a minimum 1-hour rest period between flights

The solution also includes performance profiling and visual visualizations.

🧠 Problem Motivation

Airlines must optimize which crew members operate which flights while avoiding conflicts in schedules.
This real-world task is computationally complex because:

As the number of flights increases, possible schedule combinations grow exponentially.

Therefore, crew scheduling is commonly solved using optimization and search algorithms — here we use backtracking for exact assignment.

🔍 Core Features

✔ Constraint checking (overlap + minimum rest time)
✔ Backtracking assignment algorithm
✔ Time & memory profiling using time and tracemalloc
✔ Recursive call counter to measure search complexity
✔ Gantt chart visualization showing which crew operates which flights
✔ Scaling experiment: execution time vs number of flights

📂 File Information
File	Description
crew_scheduling.py	Main executable Python script
/mnt/data/crew_assignment.json	Output file storing best schedule
/mnt/data/crew_gantt.png	Gantt chart of crew vs flights
/mnt/data/scaling_time.png	Plot of time vs flight count
▶ How to Run
Using terminal / command prompt
python crew_scheduling.py

Output generated automatically

Prints assigned crew per flight

Saves the schedule as JSON

Exports charts to images

🏗 How the Algorithm Works

Flights and crew members are defined

Flights are assigned one by one

For each flight, the algorithm tries assigning it to every crew member

If a conflict occurs → undo and try a different path

Continue until:

all flights are assigned (success) OR

no valid assignment exists

This strategy guarantees a correct solution but requires exponential search time as problem size increases.

📊 Profiling & Observations
Factor	Result
Performance grows with flights	Exponential
Memory usage	Small to moderate
Most expensive operation	Backtracking recursion

The scaling experiment visually shows how runtime increases as the number of flights becomes larger.

🧾 Conclusion

This project demonstrates how backtracking can be used to solve crew scheduling, a real-world optimization problem.
However, due to exponential growth, practical airline scheduling requires advanced techniques such as:

Branch and bound

Heuristics

Dynamic programming

AI / Genetic algorithms

Still, this implementation provides a strong foundation in:

Constraint satisfaction problems (CSP)

Combinatorial search

Performance profiling

🙌 Acknowledgments

This project reinforces concepts learned in:

Design and Analysis of Algorithms

Time–space complexity

Constraint-based optimization
