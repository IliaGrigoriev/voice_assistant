function tokenClass(token) {
  if (/^".*"\s*:$/.test(token)) return "json-key";
  if (/^"/.test(token)) return "json-string";
  if (/^(true|false)$/.test(token)) return "json-boolean";
  if (token === "null") return "json-null";
  if (/^-?\d/.test(token)) return "json-number";
  return "";
}

export default function JsonView({ data }) {
  const json = JSON.stringify(data, null, 2);
  const tokens = json.split(
    /("(?:\\.|[^"\\])*"\s*:|"(?:\\.|[^"\\])*"|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?)/g,
  );

  return (
    <div className="json-view">
      <pre>
        <code>
          {tokens.map((token, index) => (
            <span className={tokenClass(token)} key={`${index}-${token}`}>
              {token}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
