import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "請告訴我展碁國際的公司概況",
        value: "請告訴我展碁國際的公司概況"
    },
    { text: "展碁國際的核心價值?", value: "展碁國際的核心價值?" },
    { text: "展碁國際的負責人是?", value: "展碁國際的負責人是?" }
];


interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
