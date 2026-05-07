with open('croissant.json', 'r', encoding='utf-8') as f:
    text = f.read()
new_text = text.replace('Ali-Iranmanesh/HearSay', 'Aliiiranmanesh/hearSay')
with open('croissant.json', 'w', encoding='utf-8') as f:
    f.write(new_text)

with open('metadata.json', 'r', encoding='utf-8') as f:
    text = f.read()
new_text = text.replace('Ali-Iranmanesh/HearSay', 'Aliiiranmanesh/hearSay')
with open('metadata.json', 'w', encoding='utf-8') as f:
    f.write(new_text)

with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()
new_text = text.replace('Ali-Iranmanesh/HearSay', 'Aliiiranmanesh/hearSay')
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(new_text)

print("SUCCESS: Updated all files with correct username: Aliiiranmanesh/hearSay")
