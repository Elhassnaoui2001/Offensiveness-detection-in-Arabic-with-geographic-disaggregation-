# import streamlit as st



# # Team members information
# team_members = [
#     {"name": "Mohamed Elhassnaoui", "role": "Data Scientist", "linkedin": "https://www.linkedin.com/in/mohamed-elhassnaoui-7a2162211/"},
#     {"name": "Salhi Abderrahmane", "role": "Data Scientist", "linkedin": "https://www.linkedin.com/in/abderrahman-salhi-46b408253/"}
# ]

# def main():
#     # Streamlit app title
#     st.title("Team Members")

#     # Display team members information
#     for member in team_members:
#         st.subheader(member["name"])
#         st.write(f"Role: {member['role']}")
#         st.write(f"LinkedIn: [{member['name']}'s LinkedIn Profile]({member['linkedin']})")
#         st.write("\n---")

# if __name__ == "__main__":
#     main()
import streamlit as st

# Team members information with LinkedIn profiles and image filenames
team_members = [
    {"name": "Mohamed Elhassnaoui", "role": "Data Scientist", "linkedin": "https://www.linkedin.com/in/your_linkedin_profile", "image": "mohamed_image.jpg"},
    {"name": "Salhi Abderrahmane", "role": "Data Scientist", "linkedin": "https://www.linkedin.com/in/your_linkedin_profile", "image": "salhi_image.jpg"}
]

def main():
    # Streamlit app title
    st.title("Team Work Presentation")

    # Display team members information with images
    st.header("Team Members")
    for member in team_members:
        st.subheader(member["name"])
        image_path = f"images/{member['salhi.jpg']}"  # Assuming the images are in the 'images' folder
        st.image(image_path, caption=f"{member['name']} - {member['role']}", use_column_width=True)
        st.write(f"Role: {member['role']}")
        st.write(f"LinkedIn: [{member['name']}'s LinkedIn Profile]({member['linkedin']})")
        st.write("\n---")

if __name__ == "__main__":
    main()
