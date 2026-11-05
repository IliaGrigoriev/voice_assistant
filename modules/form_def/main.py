from form import Form

def a():
    form = Form("form.yaml")

    form.fill("load", path="input.csv")
    form.fill("transform", operation="normalize")

    print(form.missing())
    # {'save': ['destination']}

    form.fill("save", destination="output.csv")

    print(form.is_complete())
    # True

    print(form.get("save"))
    # {'destination': 'output.csv'}

if __name__ == "__main__":
    a()