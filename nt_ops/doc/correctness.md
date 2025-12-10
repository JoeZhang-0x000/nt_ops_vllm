This document aims to represent the correctness of the Ninetoothed operators.

## Original VLLM
Just remove the `import nt_ops` in the `nt_ops_vllm/examples/basic.py` and run the example. We will get the expected output.

```
Generated Outputs:
------------------------------------------------------------
Prompt:    'Hello, my name is'
Output:    " Josh and I'm in the middle of a project to develop a hybrid mobile app. I'm looking for guidance on how to go about using modular frameworks. I want to use React and Vue. I need to decide on the framework to use. Can you help me choose the right framework and suggest some best practices for using them?\n\nAdditionally, I want to know what are the best practices for using a modular framework in the context of a web application? Also, what are the best practices for using a"
------------------------------------------------------------
Prompt:    'The president of the United States is'
Output:    ' the most powerful individual in the world, but what about the most powerful individual in the world who is also a woman? The most powerful individual in the world who is also a woman, especially in the second half of the 20th century. The president of the United States is the most powerful individual in the world. This is the first time in history that a woman president has been elected to the presidency. The only question is, what is the most powerful individual in the world who is also'
------------------------------------------------------------
Prompt:    'The capital of France is'
Output:    ' Paris. Therefore, the capital of Paris is Paris. This is an example of __________.\n\nThe correct answer is: [list]\n\nOptions: A. A fallacy of composition\n\nB. A fallacy of composition\n\nC. A fallacy of division\n\nD. A fallacy of definition\n\nAnswer: \\boxed{C}\n\nExplanation: The capital of France is Paris, and Paris is the capital of France. Therefore, the capital of Paris is Paris. This is an example of a'
------------------------------------------------------------
Prompt:    'The future of AI is'
Output:    ' becoming more interesting and plausible, especially in the areas of the fusion of AI with other technologies. The study of AI and its development is evolving rapidly, and researchers are exploring new ways to implement AI in existing systems, including AI systems that are powered by neural networks. These systems are becoming more popular as they offer advanced capabilities. However, there is a lot of controversy and debate regarding their use in various sectors, including healthcare, education, and manufacturing. For example, in the healthcare sector, AI can'
------------------------------------------------------------
```
## With NT
```
Generated Outputs:
------------------------------------------------------------
Prompt:    'Hello, my name is'
Output:    " Josh Eurasian. I'm very curious to know if there are any healthy ways to improve the health and life quality of the elderly people, and how to help them to live better in the future. I'm looking for the steps that are effective for this. I don't want to make the whole world better, but rather to help the elderly people to live better. Thank you very much for your help. Thank you again for your kind response!\nHello! Well, I understand your concern about improving"
------------------------------------------------------------
Prompt:    'The president of the United States is'
Output:    ' the most powerful individual in the world, but what about the most powerful individual in the world who is also a woman? The most powerful individual in the world who is also a woman, especially in the second half of the 20th century. The president of the United States is the most powerful individual in the world. What about the other female leaders who are also the most powerful individuals? The answer is that they are not powerful, but the most powerful individual in the world is the president of'
------------------------------------------------------------
Prompt:    'The capital of France is'
Output:    ' Paris. Therefore, the capital of Paris is Paris. This is an example of __________.\n\nThe correct answer is: [list]\n\nOptions: A. A fallacy of composition\n\nB. A fallacy of composition\n\nC. A fallacy of division\n\nD. A fallacy of definition\n\nAnswer: \\boxed{C}\n\nExplanation: The capital of France is Paris, and Paris is the capital of France. Therefore, the capital of Paris is Paris. This is an example of a'
------------------------------------------------------------
Prompt:    'The future of AI is'
Output:    ' becoming more interesting and plausible, especially in the areas of the fusion of AI with other technologies. The study of AI and its development is evolving rapidly, and researchers are exploring new ways to implement it in existing systems, including AI systems that are powered by neural networks. These systems are becoming more popular as they offer advanced capabilities. However, there is a lot of controversy and debate regarding their use in various sectors, including healthcare, education, and manufacturing. For example, in the healthcare sector, AI can'
```