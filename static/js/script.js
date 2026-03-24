async function analyze() {
    const text = document.getElementById("inputText").value;

    const res = await fetch("/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(text)
    });

    const data = await res.json();

    document.getElementById("result").innerText =
        JSON.stringify(data, null, 2);
}