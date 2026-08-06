from backend import run_travel_agent

user_input = input("Enter travel request: ")

response = run_travel_agent(
    user_input=user_input,
    thread_id="test_user"
)

print("\n--- FLIGHT RESULTS (raw) ---\n")
print(response["flight_offers"])

print("\n--- LIVE FLIGHT RESULTS (raw) ---\n")
print(response["live_status"])

print("\n--- HOTEL RESULTS (raw) ---\n")
print(response["hotel_results"])

print("\n--- ITINERARY (raw) ---\n")
print(response["itinerary"])

print("\n--- FINAL RESPONSE ---\n")
print(response["answer"])

# print(f"\n(LLM calls used: {response['llm_calls']})")