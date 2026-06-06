# creating the base agent from which all the other agents will inherit the basic architecture 

import json,csv,os
from datetime import datetime

class BaseAgent:
    def __init__(self,name):
        self.name = name
        self.short_term_memory = {}
        self.long_term_memory = {}
        self.episode_count = 0
        os.makedirs("memory", exist_ok=True)
        self.memory_path = f"memory/{name}_memory.json"
        self.log_path = f"memory/{name}_log.csv"
        self.load_memory()
        self.init_log()
        print(f"{self.name}: Past episodes are {self.episode_count}")

#make function to load memory in agent
    def load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path) as f:
                data = json.load(f)
                self.long_term_memory = data.get("memory", {})
                self.episode_count = data.get("episode_count", 0)    

 #make function to save memory
    def save_memory(self):
        with open(self.memory_path, "w") as f:
            json.dump({"agent": self.name, "episode_count": self.episode_count,"last_updated": datetime.now().isoformat(),"memory": self.long_term_memory},f, indent=2,default=str)

#make function to remember memory
    def remember(self, key, value):
        self.long_term_memory[key] = value
        self.save_memory()

#make function to recall what is learnt
    def recall(self, key, default=None):
        return self.long_term_memory.get(key, default)

#make function to create log file to store all episodes" details
    def init_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["episode","timestamp","action","input_summary","output_summary","reward"])           

#amke function to make log after every episode
    def log(self, action, inp, out, reward=None):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([self.episode_count,datetime.now().isoformat(), action,str(inp)[:300], str(out)[:300],reward])


#def observe think act here but there task will be written separately for different agents
    def observe(self, data):
        raise NotImplementedError
    def think(self):
        raise NotImplementedError
    def act(self):
        raise NotImplementedError
 
    def run_episode(self, data):
        self.episode_count += 1
        self.observe(data)
        self.think()
        self.save_memory()
        return self.act()            