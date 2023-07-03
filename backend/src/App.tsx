import ListGroup from "./components/ListGroup";

function App() {
  let items = ["Paris", "London", "Tokyo", "Seoul"];
  const handleSelectItem = (item:string) =>{console.log(item)}
  return (
    <div>
      <ListGroup items={items} title="Cities" onSelectItem={handleSelectItem} />
    </div>
  );
}
export default App;
