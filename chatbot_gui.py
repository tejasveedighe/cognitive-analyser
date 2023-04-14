import tkinter as tk
import csv

class ChatbotGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chatbot")

        # Create a text box for the chatbot's messages
        self.message_box = tk.Text(self.root, height=30, width=100)
        self.message_box.pack(side=tk.TOP, padx=10, pady=10)

        # Create a text entry field for the user's responses
        self.user_input = tk.Entry(self.root ,width=100)
        self.user_input.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Bind the user input field to the Enter key
        self.user_input.bind("<Return>", self.handle_user_input)

        # Start the chatbot conversation
        self.greet_user()
        self.current_question_index = 0
        self.questions = ["Hi What is your name?"]
        self.user_responses = {}

        self.ask_question(self.questions[self.current_question_index])

        self.root.mainloop()

    def greet_user(self):
        # Display the chatbot's greeting in the message box
        self.message_box.insert(tk.END, "Chatbot: Hi! I'm a chatbot.")

    def ask_question(self, message):
        # Display the chatbot's message in the message box
        self.message_box.insert(tk.END, "Chatbot: " + message + "\n")

    def handle_user_input(self, event):
        # Get the user's response and clear the input field
        user_response = self.user_input.get()
        self.user_input.delete(0, tk.END)

        # Check that the user's response has a length greater than 3
        if len(user_response) <= 3:
            self.message_box.insert(tk.END, "Please enter a longer response.\n")
            return

        # Display the user's response in the message box
        self.message_box.insert(tk.END, "You: " + user_response + "\n")

        # Save the user's response to the user_responses dictionary
        self.user_responses[self.current_question_index] = user_response

        # Determine the chatbot's response based on the user's input
        if self.current_question_index == 0:
            # Assume that the user's name is the first word in their response
            self.user_name = user_response.split()[0]
            self.current_question_index += 1
            self.questions.append(f"Nice to meet you, {self.user_name}! " +
                                  "What is your profession? (student, businessman, worker)")
            self.ask_question(self.questions[self.current_question_index])
        elif self.current_question_index == 1:
            self.user_profession = user_response.lower()
            if self.user_profession == "student":
                self.questions.extend(['How do you balance your studies and personal life?',
                                       'Do you feel overwhelmed with schoolwork or exams?',
                                       'Do you have a good support system to help you manage stress?',
                                       "Do you feel comfortable talking about your mental health with someone at school (e.g., counselor, teacher, coach)?",
                                       "Have you had trouble sleeping recently?",
                                       "How often do you engage in physical activity or exercise?",
                                       "Do you feel like you have a good support system of friends and family?",
                                       "Have you experienced any feelings of hopelessness or worthlessness in the past month?",
                                       "Do you have any concerns about your mental health that you would like to discuss with a professional?",
                                       "How much stress do you feel related to your academic workload?",
                                       "Have you experienced any negative emotions related to school or academic performance, such as frustration or low self-esteem?",
                                       "How many hours per day do you typically spend on schoolwork or studying?",
                                       "Do you feel like you have a healthy balance between schoolwork and other activities?",
                                       "Have you been participating in any extracurricular activities or hobbies that you enjoy?",
                                       "Have you noticed any changes in your motivation or interest in school or activities?",
                                       "Have you been feeling socially isolated or disconnected from others at school or in your community?",
                                       "Have you experienced any conflicts or negative experiences with peers or authority figures at school?",
                                       "Do you feel like you have access to adequate resources and support to succeed academically?",
                                       "Do you feel like your mental health is being adequately supported by your school or academic environment?"
                                    ]
                                      )
            elif self.user_profession == "businessman":
                self.questions.extend(['Do you feel stressed with your work responsibilities?',
                                       'Do you take breaks during your workday?',
                                       'How do you manage your work-life balance?',
                                       "How do you manage stress in your day-to-day life as a businessman?",
                                       "What are some warning signs of burnout or depression that you have experienced in the past, and how did you address them?",
                                       "Have you sought out any mental health resources or support in the past? If so, what was your experience like?",
                                       "What strategies do you use to maintain a healthy work-life balance?",
                                       "Have you ever experienced anxiety or panic attacks related to work? If so, how did you cope with them?",
                                       "How do you prioritize self-care and relaxation in your busy schedule?",
                                       "Have you ever experienced feelings of inadequacy or imposter syndrome as a businessman? If so, how did you overcome them?",
                                       "Have you ever encountered ethical dilemmas or conflicts in your business that have affected your mental health? If so, how did you manage them?",
                                       "What steps do you take to manage difficult conversations or conflicts with colleagues or clients in a healthy and constructive way?",
                                       "Do you have any advice for other businessmen who may be struggling with their mental health?"
                                       ])
            elif self.user_profession == "worker":
                self.questions.extend(['Do you have a good work-life balance?',
                                       'Do you feel supported by your colleagues and supervisors?',
                                       'Do you feel stressed with your workload?',
                                       "How do you prioritize self-care and stress management in your profession?",
                                       "Have you ever experienced burnout or compassion fatigue in your work? If so, how did you address it?",
                                       "What strategies do you use to maintain a healthy work-life balance?",
                                       "Have you ever experienced discrimination or harassment in your workplace, and how did it affect your mental health?",
                                       "What resources or support have you sought out for your mental health, and how have they helped?",
                                       "How do you manage anxiety or other mental health challenges while performing your job duties?",
                                       "Have you ever encountered ethical dilemmas or conflicts in your profession that have affected your mental health?, If so, how did you manage them?",
                                       "How do you handle high-pressure situations or intense deadlines without negatively impacting your mental health?",
                                       "Have you ever struggled with imposter syndrome or feelings of inadequacy in your profession? If so, how did you overcome them?",
                                       "Do you have any advice for others in your profession who may be struggling with their mental health?,"
                                       ])
            self.current_question_index += 1
            self.ask_question(self.questions[self.current_question_index])
        elif self.current_question_index < len(self.questions) - 1:
            self.current_question_index += 1
            self.ask_question(self.questions[self.current_question_index])
        else:
            self.ask_question("Thanks for chatting with me!")
            self.export_user_responses()
            self.root.quit()

    def export_user_responses(self):
        # Write the user's responses to a CSV file
        with open(f"user_responses.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Response"])
            for question_index, response in self.user_responses.items():
                writer.writerow([response])

        self.message_box.insert(
            tk.END, f"Your responses have been saved to {self.user_name}_responses.csv\n")


if __name__ == "__main__":
    ChatbotGUI()
